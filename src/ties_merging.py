import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())


import wandb
import argparse
import logging
import torch
import numpy as np
import subprocess
import json

from tqdm import tqdm
from src.data.dataset_mixtures import (
    DATASET_MIXTURES,
    get_dataset_from_argparse,
    get_datasetMixture,
)


from src.train.TrainingConfig import TrainingConfig

from src.model.ModelConfig import ModelConfig
from src.model.load_model import (
    load_model,
    construct_model,
    get_pretrained_model,
    load_checkpoints,
    merge_checkpoints,
    loadCheckpoint_intoModel,
    check_parameterNamesMatch,
)

from src.model.utils import get_parameterCount
from src.model.checkpoint_filepaths import (
    generate_checkpointFilepath,
    generate_MixtureCheckpointFilepath,
)

from src.data.Batcher import Batcher
from src.data.dataset_readers import get_datasetReader
from src.data.dataset_mixtures import get_datasetMixtureReader, DatasetMixtureReader
from src.data.PytorchDataset import PytorchDataset
from src.inference import inference

from src.utils.utils import (
    parse_modelName,
    ParseKwargs,
    get_logger,
    set_seeds,
    setup_wandb_logger,
)
from src.utils.merge_utils import *

logger = logging.getLogger("root")


def resolve_lambda_code(lambda_code):
    if type(lambda_code) is tuple:
        lambda_list = torch.tensor(lambda_code)
    elif isinstance(lambda_code, float) or isinstance(lambda_code, int):
        lambda_list = torch.tensor([lambda_code])
    elif "linear+" in lambda_code:
        search_lambda, start, end, step = lambda_code.split("+")
        lambda_list = np.arange(eval(start), eval(end), eval(step))
    elif "mergelist" in lambda_code:
        task_lambdas = lambda_code.split("+")[-1].split(",")
        lambda_list = np.array(task_lambdas).astype(float).tolist()
    else:
        raise NotImplementedError(f"Unable to decode lambda_code {lambda_code}")
    return lambda_list


def merge_and_evalaute(
    config_toInit,
    merge_function,
    dataset_mixture_to_merge,
    multiple_prompts,
    all_inference_dataset_mixtures,
    # tracker,
    inference_kwargs,
    device,
):
    # Load pretrained model.
    model_config = ModelConfig(
        configDict_toInitializeFrom=config_toInit.get_dict(),
    )
    model, tokenizer, peftParameter_regex = construct_model(
        model_config.pretrained_model,
        model_config.peft_method,
        model_config.max_seq_len,
        device="cpu",
        model_config=model_config,
    )

    logger.info(f"Loading Checkpoints for Merging: {config_toInit.pretrained_model}")
    all_mixing_datasets = get_dataset_from_argparse(dataset_mixture_to_merge)
    dataset_folder = ",".join(d for d in all_mixing_datasets)
    if config_toInit.pretrained_model == "bigscience/T0_3B":
        (
            tv_flat_checks,
            flat_ptm,
            flat_ft,
            ft_checks,
            ptm_check,
            merging_tasks,
            dump_dir,
            remove_keys,
            reshape_keys,
            base_model,
            load_dir,
            filepaths,
        ) = load_ia3(all_mixing_datasets, config_toInit.pretrained_model)
    elif "t5" in config_toInit.pretrained_model:
        (
            tv_flat_checks,
            flat_ptm,
            flat_ft,
            ft_checks,
            ptm_check,
            merging_tasks,
            dump_dir,
            remove_keys,
            reshape_keys,
            base_model,
            load_dir,
            filepaths,
        ) = load_t5(
            all_mixing_datasets,
            config_toInit.pretrained_model,
            pretrained_state_dict=model.state_dict(),
        )
    else:
        raise ValueError(
            f"Merging not implemented for model: {config_toInit.pretrained_model}"
        )

    # Set model to device
    model.to(device)

    cached_dataset_reader = {}
    if "basic" in merge_function:
        logger.info(f"Performing Basic Merging with {merge_function}")
        merge_type, agg_code = merge_function.split("_")
        merged_checkpoint = basic_merging(agg_code, flat_ft, ft_checks, remove_keys)
        # Delete to clean up CPU memory
        del tv_flat_checks, flat_ptm, flat_ft, ft_checks, ptm_check

        model = loadCheckpoint_intoModel(merged_checkpoint, model)
        # Evaluate
        experiment_dir = os.path.join(
            config_toInit.experiment_dir,
            dataset_folder,
        )
        cached_dataset_reader = inference(
            model,
            tokenizer,
            config_toInit,
            model_config,
            cached_datasetReaders=cached_dataset_reader,
            across_multiplePrompts=multiple_prompts,
            experiment_dir=experiment_dir,
            all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
            inference_kwargs=inference_kwargs,
            device=device,
        )
    elif "task-vector" in merge_function:
        merge_type, lambda_code = merge_function.split("_")
        merged_tv = tv_merging(tv_flat_checks)
        # Delete to clean up CPU memory
        reference_state_dict = ft_checks[0]
        del flat_ft, ft_checks, ptm_check

        lambdas = resolve_lambda_code(lambda_code)
        for lam in lambdas:
            lam = round(lam, 1)
            logger.info(
                f"Performing Task Vector Merging with {merge_function} and Lambda {lam:.1f}"
            )
            merged_check = flat_ptm + lam * merged_tv
            merged_checkpoint = vector_to_state_dict(
                merged_check, reference_state_dict, remove_keys=remove_keys
            )
            model = loadCheckpoint_intoModel(merged_checkpoint, model)
            # Evaluate
            experiment_dir = os.path.join(
                config_toInit.experiment_dir,
                dataset_folder,
                f"lambda_{lam:.1f}",
            )
            cached_dataset_reader = inference(
                model,
                tokenizer,
                config_toInit,
                model_config,
                cached_datasetReaders=cached_dataset_reader,
                across_multiplePrompts=multiple_prompts,
                experiment_dir=experiment_dir,
                all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
                inference_kwargs=inference_kwargs,
                device=device,
            )
    elif "oracle" in merge_function:
        oracle, reset, merge, lambda_code = merge_function.split("_")
        if "topk" in reset:
            reset_type = "topk"
            reset_thresh = eval(reset[len(reset_type) :])
        elif "std" in reset:
            reset_type = "std"
            reset_thresh = eval(reset[len(reset_type) :])
        elif "nf" in reset:
            reset_type = "nf"
            reset_thresh = eval(reset[len(reset_type) :])
        else:
            reset_type = ""
            reset_thresh = "none"

        logger.info(
            f"Loading Multitask checkpoint from {MULTITASK_PATH[model_config.pretrained_model]}"
        )
        multitask_check = torch.load(MULTITASK_PATH[model_config.pretrained_model])
        multitask_flat_tv = state_dict_to_vector(multitask_check, remove_keys) - 1
        multitask_sign = multitask_flat_tv.sign()
        assert multitask_sign.shape[-1] == tv_flat_checks.shape[-1]
        merged_tv = merge_with_oracle_sign(
            multitask_sign, tv_flat_checks, reset_thresh, merge
        )

        # Delete to clean up CPU memory
        reference_state_dict = ft_checks[0]
        del flat_ft, ft_checks, ptm_check

        lambdas = resolve_lambda_code(lambda_code)
        for lam in lambdas:
            lam = round(lam, 1)
            logger.info(
                f"Performing PRESS Merging with ORACLE SIGNS using {merge_function} and Lambda {lam:.1f}"
            )
            merged_check = flat_ptm + lam * merged_tv
            merged_checkpoint = vector_to_state_dict(
                merged_check, reference_state_dict, remove_keys=remove_keys
            )
            model = loadCheckpoint_intoModel(merged_checkpoint, model)
            # Evaluate
            experiment_dir = os.path.join(
                config_toInit.experiment_dir,
                dataset_folder,
                f"lambda_{lam:.1f}",
            )
            cached_dataset_reader = inference(
                model,
                tokenizer,
                config_toInit,
                model_config,
                cached_datasetReaders=cached_dataset_reader,
                across_multiplePrompts=multiple_prompts,
                experiment_dir=experiment_dir,
                all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
                inference_kwargs=inference_kwargs,
                device=device,
            )
    else:
        reset, resolve, merge, lambda_code = merge_function.split("_")
        if "topk" in reset:
            reset_type = "topk"
            reset_thresh = eval(reset[len(reset_type) :])
        elif "std" in reset:
            reset_type = "std"
            reset_thresh = eval(reset[len(reset_type) :])
        elif "nf" in reset:
            reset_type = "nf"
            reset_thresh = eval(reset[len(reset_type) :])
        else:
            reset_type = ""
            reset_thresh = "none"

        merged_tv = merge_methods(
            reset_type,
            tv_flat_checks,
            reset_thresh=reset_thresh,
            resolve_method=resolve,
            merge_func=merge,
        )
        # Delete to clean up CPU memory
        reference_state_dict = ft_checks[0]
        del flat_ft, ft_checks, ptm_check

        lambdas = resolve_lambda_code(lambda_code)
        for lam in lambdas:
            lam = round(lam, 1)
            logger.info(
                f"Performing PRESS Merging with {merge_function} and Lambda {lam:.1f}"
            )
            merged_check = flat_ptm + lam * merged_tv
            merged_checkpoint = vector_to_state_dict(
                merged_check, reference_state_dict, remove_keys=remove_keys
            )
            model = loadCheckpoint_intoModel(merged_checkpoint, model)
            # Evaluate
            experiment_dir = os.path.join(
                config_toInit.experiment_dir,
                dataset_folder,
                f"lambda_{lam:.1f}",
            )
            cached_dataset_reader = inference(
                model,
                tokenizer,
                config_toInit,
                model_config,
                cached_datasetReaders=cached_dataset_reader,
                across_multiplePrompts=multiple_prompts,
                experiment_dir=experiment_dir,
                all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
                inference_kwargs=inference_kwargs,
                device=device,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_to_init", type=str, required=True)
    parser.add_argument("-i", "--all_inference_dataset_mixtures", type=str, nargs="*")
    parser.add_argument("-m", "--merging_datasets_or_mixtures", type=str, nargs="*")
    parser.add_argument("--multiple_prompts", action="store_true")
    parser.add_argument(
        "--kwargs",
        nargs="*",
        action=ParseKwargs,
        default={
            "split": "validation",
            "should_save_to_gcp": False,
            "world_size": None,
            "eval_template_idx": -1,
        },
    )
    # basic_[mean, median, magnitude], task-vectors_[xx, linear+0+1+0.1, mergelist]
    # [none, topkx]_[mass, normfrac]_[mean, sum, median, magnitude, dis-mean, dis-sum]_[none, xx, linear+0+1+0.1, mergelist]
    parser.add_argument("-f", "--merge_function", type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_toInit = TrainingConfig(
        config_filepaths=[args.config_to_init], kwargs=args.kwargs, create_expDir=True
    )

    log_config_path = os.path.join(os.getcwd(), "src", "utils", "log_config.json")
    logger = get_logger("log.txt", f"{config_toInit.experiment_dir}/", log_config_path)
    logger.info("Starting Experiments of Model merging")

    command = subprocess.list2cmdline(["python"] + sys.argv)
    logger.info(command)

    merge_and_evalaute(
        config_toInit,
        args.merge_function,
        args.merging_datasets_or_mixtures[0].split(","),
        args.multiple_prompts,
        args.all_inference_dataset_mixtures[0].split(","),
        args.kwargs,
        device,
    )
