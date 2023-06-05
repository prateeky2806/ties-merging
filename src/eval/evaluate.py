from tqdm import tqdm
import logging
import torch.distributed as dist

from src.data.dataset_readers import get_datasetReader
from src.data.Batcher import Batcher
from src.data.PytorchDataset import PytorchDataset

from src.eval.utils import prepare_batchOfEvalInfo, getAndMake_specificPredictionDir
from src.eval.cache import getCached_predictions, does_cachePredictionExist
from src.eval.Evaluator import Evaluator


from src.utils.distributed_utils import (
    reduce_gatheredOutput,
    is_nodeZero,
    is_distributedSetup,
)

import torch

logger = logging.getLogger("root")


def evaluate_model(
    model, batch_iterator, metrics, evaluation_config, specificPrediction_dir, device
):
    """

    Args:
        model:
        tokenizer:
        batch_iterator:
        metrics:
        evaluation_config:
        specificPrediction_dir:
        device:

    Returns:

    """
    logger.info(f"\tEvaluating model on {evaluation_config.inference_dataset} dataset")

    if is_nodeZero(device):
        (
            canUseCached_predictionFP,
            cached_evaluationConfigDict_toUse,
            prediction_fp,
        ) = does_cachePredictionExist(
            evaluation_config,
            specificPrediction_dir,
            keys_mustMatch=[
                "use_bfloat16_during_eval",
                "max_gen_len",
                "few_shot_random_seed",
            ],
        )

        if canUseCached_predictionFP:
            return getCached_predictions(
                evaluation_config,
                cached_evaluationConfigDict_toUse,
                prediction_fp,
                metrics,
            )

        # evaluator must be created after checking if the cache of model predictions exists,
        # since evaluator creates the prediction fp
        evaluator = Evaluator(
            metrics, prediction_fp, evaluation_config.should_save_to_gcp
        )

    model.eval()

    # When using DDP model, the model is wrapped with a DDP model and must be called with
    # model.module
    if is_distributedSetup(evaluation_config.world_size):
        model = model.module

    with torch.no_grad():
        for batch in tqdm(batch_iterator):

            batchOf_evalInfo = prepare_batchOfEvalInfo(batch)

            if "Accuracy" in metrics:
                if evaluation_config.use_bfloat16_during_eval:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        (
                            predicted_choice,
                            score_ofChoices,
                            logProbs_ofAllChoicesIds,
                            len_allChoices,
                        ) = model.predict_mulChoice(
                            batch, evaluation_config.length_normalization
                        )
                else:
                    (
                        predicted_choice,
                        score_ofChoices,
                        logProbs_ofAllChoicesIds,
                        len_allChoices,
                    ) = model.predict_mulChoice(
                        batch, evaluation_config.length_normalization
                    )

                batchOf_evalInfo.update(
                    {
                        "predicted_choice": predicted_choice,
                        "score_of_choices": score_ofChoices,
                        "log_probs_of_all_choices_ids": logProbs_ofAllChoicesIds,
                        "len_all_choices": len_allChoices,
                    }
                )

            if "Squad" in metrics:
                if evaluation_config.use_bfloat16_during_eval:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        generated_ids, generated_txt = model.generate(
                            batch, evaluation_config.max_gen_len
                        )
                else:
                    generated_ids, generated_txt = model.generate(
                        batch, evaluation_config.max_gen_len
                    )

                batchOf_evalInfo.update(
                    {"generated_ids": generated_ids, "prediction_text": generated_txt}
                )

            if is_distributedSetup(evaluation_config.world_size):
                gathered_batchOfEvalInfo = [{}] * evaluation_config.world_size

                dist.gather_object(
                    batchOf_evalInfo,
                    gathered_batchOfEvalInfo if is_nodeZero(device) else None,
                    dst=0,
                )

                if is_nodeZero(device):
                    batchOf_evalInfo = reduce_gatheredOutput(gathered_batchOfEvalInfo)

            if is_nodeZero(device):
                evaluator.add_batch(batchOf_evalInfo)

    if is_nodeZero(device):
        return evaluator.get_result()
    else:
        return None


def evaluate_fromConfig(
    model, tokenizer, cached_datasetReaders, evaluation_config, device
):
    """

    Args:
        model:
        tokenizer:
        cached_datasetReaders:
        evaluation_config:
        device:

    Returns:
        scores:
        cached_datasetReaders
    """

    if evaluation_config.inference_dataset in cached_datasetReaders:
        dataset_reader = cached_datasetReaders[evaluation_config.inference_dataset]
    else:
        dataset_kwargs = {
            "few_shot_random_seed": evaluation_config.few_shot_random_seed,
            "num_val_samples": evaluation_config.num_val_samples,
            "max_datapoints_per_dataset_without_templates": evaluation_config.max_datapoints_per_dataset_without_templates,
        }
        dataset_reader = get_datasetReader(
            evaluation_config.inference_dataset, dataset_kwargs=dataset_kwargs
        )
        cached_datasetReaders[evaluation_config.inference_dataset] = dataset_reader

    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, device)

    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=None,
        eval_batchSize=evaluation_config.eval_batch_size,
        world_size=evaluation_config.world_size,
        device=device,
    )
    getBatches_fn = lambda batcher, template_idx: batcher.get_evalBatches(
        evaluation_config.split, template_idx
    )

    # Only node zero should get the prediction dir in case the prediction dir
    # doesn't exist and it must be created in getAndMake_specificPredictionDir
    if is_nodeZero(device):
        specificPrediction_dir = getAndMake_specificPredictionDir(
            evaluation_config.prediction_dir,
            evaluation_config.split,
            evaluation_config.inference_dataset,
            evaluation_config.eval_template_idx,
        )
    else:
        specificPrediction_dir = None

    return (
        evaluate_model(
            model,
            getBatches_fn(batcher, evaluation_config.eval_template_idx),
            batcher.get_metricsForDataset(),
            evaluation_config,
            specificPrediction_dir,
            device,
        ),
        cached_datasetReaders,
    )


def evaluate_multipleConfigs(
    model, tokenizer, cached_datasetReaders, multiEvaluation_config, device
):
    """

    Args:
        model:
        tokenizer:
        cached_datasetReaders:
        multiEvaluation_config:
        device:

    Returns:
        multiple_ConfigAndScores:
        cached_datasetReaders:
    """
    if is_nodeZero(device):
        multiple_ConfigAndScores = []

    for evaluation_config in multiEvaluation_config.get_allConfigs():
        scores, cached_datasetReaders = evaluate_fromConfig(
            model, tokenizer, cached_datasetReaders, evaluation_config, device
        )
        if is_nodeZero(device):
            multiple_ConfigAndScores.append(
                {"config": evaluation_config.get_dict(), "score": scores}
            )

    if is_nodeZero(device):
        return multiple_ConfigAndScores, cached_datasetReaders
    else:
        return None, cached_datasetReaders
