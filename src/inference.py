import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())


import argparse
import logging
import torch
import json

from src.train.TrainingConfig import TrainingConfig
from src.data.dataset_readers import DATASET_CLASSES
from src.data.dataset_mixtures import DATASET_MIXTURES, get_dataset_from_argparse

from src.model.load_model import load_model
from src.model.ModelConfig import ModelConfig
from src.model.checkpoint_filepaths import (
    generate_checkpointFilepath,
)

from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.MultiEvaluationConfig import MultiEvaluationConfig
from src.eval.evaluate import evaluate_multipleConfigs, evaluate_fromConfig
from src.eval.scores import (
    concatenate_scores,
    get_averageScores,
    get_summaryOfScores_acrossPrompts,
    save_allScores,
    saveResult_acrossDatasetMixture,
)

from src.data.dataset_mixtures import get_datasetMixture
from src.data.dataset_readers import get_datasetReader

from src.utils.distributed_utils import is_nodeZero
from src.utils.utils import (
    ParseKwargs,
    get_average,
    group_by,
    map_forDictionaries,
    deep_update,
    get_logger,
)

logger = logging.getLogger("root")


def inference_withMultiplePrompts(
    model,
    tokenizer,
    model_config,
    evaluation_config,
    cached_datasetReaders,
    inference_dataset_mixture,
    score_fp,
    device,
):

    values_toIterateOver = {}

    if inference_dataset_mixture is not None:
        all_datasets = get_datasetMixture(inference_dataset_mixture)
    else:
        assert evaluation_config.inference_dataset is not None
        all_datasets = evaluation_config.inference_dataset

    for dataset_name in all_datasets:
        dataset_kwargs = {
            "few_shot_random_seed": evaluation_config.few_shot_random_seed,
            "num_val_samples": evaluation_config.num_val_samples,
            "max_datapoints_per_dataset_without_templates": evaluation_config.max_datapoints_per_dataset_without_templates,
        }
        dataset_reader = get_datasetReader(dataset_name, dataset_kwargs)
        cached_datasetReaders[dataset_name] = dataset_reader

        num_templates = dataset_reader.get_numTemplates()
        values_toIterateOver[dataset_name] = list(range(num_templates))

    multiEvaluation_config = MultiEvaluationConfig(
        fields_toIterateOver=["inference_dataset", "eval_template_idx"],
        values_toIterateOver=values_toIterateOver,
        configDict_toInitializeFrom=evaluation_config.get_dict(),
        fields_toUpdate={"inference_dataset": None, "eval_template_idx": None},
    )

    multiple_configAndScores, cached_datasetReaders = evaluate_multipleConfigs(
        model, tokenizer, cached_datasetReaders, multiEvaluation_config, device
    )

    if is_nodeZero(device):

        if inference_dataset_mixture != None:
            groupScores_byDataset = group_by(
                multiple_configAndScores, lambda x: x["config"]["inference_dataset"]
            )
            summaryOfScores_perDataset = map_forDictionaries(
                my_dict=groupScores_byDataset, map_fn=get_summaryOfScores_acrossPrompts
            )
            concatentedScores_perDataset = map_forDictionaries(
                my_dict=groupScores_byDataset, map_fn=concatenate_scores
            )

            multiple_prompt_scores = deep_update(
                summaryOfScores_perDataset, concatentedScores_perDataset
            )

        else:
            summaryOfScores_perDataset = get_summaryOfScores_acrossPrompts(
                multiple_configAndScores
            )
            concatentedScores_perDataset = concatenate_scores(multiple_configAndScores)
            multiple_prompt_scores = deep_update(
                summaryOfScores_perDataset, concatentedScores_perDataset
            )

        averageScore_acrossDataset = get_average(
            list(
                map(
                    lambda dataset: summaryOfScores_perDataset[dataset]["median"],
                    summaryOfScores_perDataset.keys(),
                )
            )
        )

        multiple_prompt_scores.update(
            {
                "config": multiEvaluation_config.get_dict(),
                "model_config": model_config.get_dict(),
                "average": averageScore_acrossDataset,
            }
        )

        save_allScores(multiple_prompt_scores, score_fp + ".json")

        def getScore_fn(dataset_score):
            return (
                f"{round(dataset_score['median'] * 100, 1)} "
                f"({round(dataset_score['interquartile_range'] * 100, 1)})"
            )

        saveResult_acrossDatasetMixture(
            inference_dataset_mixture,
            multiple_prompt_scores,
            getScore_fn,
            score_fp + ".txt",
            saveAverage_acrossDatasets=True,
        )

    return cached_datasetReaders


def inference_withSinglePrompt(
    model,
    tokenizer,
    model_config,
    evaluation_config,
    cached_datasetReaders,
    inference_dataset_mixture,
    score_fp,
    device,
):

    if inference_dataset_mixture is not None:
        multiEvaluation_config = MultiEvaluationConfig(
            fields_toIterateOver=["inference_dataset"],
            values_toIterateOver=None,
            configDict_toInitializeFrom=evaluation_config.get_dict(),
            fields_toUpdate={"inference_dataset": inference_dataset_mixture},
        )

        multiple_configAndScores, cached_datasetReaders = evaluate_multipleConfigs(
            model, tokenizer, cached_datasetReaders, multiEvaluation_config, device
        )

        if is_nodeZero(device):
            average_score = get_averageScores(multiple_configAndScores)
            concatenated_scores = concatenate_scores(
                multiple_configAndScores, "inference_dataset"
            )
            inference_scores = deep_update(concatenated_scores, average_score)

        config = multiEvaluation_config
    # Evaluate the original dataset in evaluation config
    else:
        assert evaluation_config.inference_dataset is not None
        inference_scores, cached_datasetReaders = evaluate_fromConfig(
            model, tokenizer, cached_datasetReaders, evaluation_config, device
        )
        config = evaluation_config

    if is_nodeZero(device):
        inference_scores["config"] = config.get_dict()
        inference_scores["model_config"] = model_config.get_dict()

        save_allScores(inference_scores, score_fp + ".json")

        def getScore_fn(dataset_score):
            return str(round(dataset_score["average"] * 100, 1))

        logger.info(f"Saving Results in {score_fp + '.txt'}")
        saveResult_acrossDatasetMixture(
            inference_dataset_mixture,
            inference_scores,
            getScore_fn,
            score_fp + ".txt",
            saveAverage_acrossDatasets=True,
        )

    return cached_datasetReaders


def inference(
    model,
    tokenizer,
    config_toInit,
    model_config,
    cached_datasetReaders,
    across_multiplePrompts,
    experiment_dir,
    all_inferenceDatasetMixtures,
    inference_kwargs,
    device,
):
    prediction_dir = os.path.join(
        experiment_dir,
        "predictions",
        f"{'multiple_prompts' if across_multiplePrompts else 'inference'}",
    )
    evaluation_config = EvaluationConfig(
        configDict_toInitializeFrom=config_toInit.get_dict(),
        fields_toUpdate={"prediction_dir": prediction_dir},
        kwargs=inference_kwargs,
    )
    score_fp = os.path.join(
        experiment_dir,
        f"{'multiple_prompt_scores' if across_multiplePrompts else 'inference_scores'}",
    )
    base_inference_fn = (
        inference_withMultiplePrompts
        if across_multiplePrompts
        else inference_withSinglePrompt
    )
    inference_fn = lambda inference_dataset_mixture: base_inference_fn(
        model,
        tokenizer,
        model_config,
        evaluation_config,
        cached_datasetReaders=cached_datasetReaders,
        inference_dataset_mixture=inference_dataset_mixture,
        score_fp=score_fp,
        device=device,
    )

    if all_inferenceDatasetMixtures is not None:
        inference_datasets = get_dataset_from_argparse(all_inferenceDatasetMixtures)
        cached_datasetReaders = inference_fn(inference_datasets)
    else:
        logger.info(
            "No inference dataset mixture passed in, so using the default inference dataset in the config."
        )
        cached_datasetReaders = inference_fn(None)

    return cached_datasetReaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_to_init", type=str, required=True)
    parser.add_argument("-i", "--all_inference_dataset_mixtures", type=str, nargs="+")
    parser.add_argument(
        "--multiple_prompts",
        action="store_true",
        help="Whether to run inference with multiple prompts and report median.",
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        action=ParseKwargs,
        default={
            "split": "validation",
            "should_save_to_gcp": False,
            "world_size": None,
        },
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_toInit = TrainingConfig(
        config_filepaths=[args.config_to_init], kwargs=args.kwargs, create_expDir=True
    )

    log_config_path = os.path.join(os.getcwd(), "src", "utils", "log_config.json")
    logger = get_logger("log.txt", f"{config_toInit.experiment_dir}/", log_config_path)
    logger.info("Starting inference!")

    model_config = ModelConfig(
        configDict_toInitializeFrom=config_toInit.get_dict(),
        kwargs=args.kwargs,
        # fields_toUpdate={"checkpoint_to_directly_load_model": checkpoint_path},
    )

    model, tokenizer, _, cached_models = load_model(
        model_config, cached_models={}, device=device
    )

    inference(
        model,
        tokenizer,
        config_toInit,
        model_config,
        cached_datasetReaders={},
        across_multiplePrompts=args.multiple_prompts,
        experiment_dir=config_toInit.experiment_dir,
        all_inferenceDatasetMixtures=args.all_inference_dataset_mixtures[0].split(","),
        inference_kwargs=args.kwargs,
        device=device,
    )
