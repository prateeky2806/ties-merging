import torch
import os
import json
import logging

logger = logging.getLogger("root")

from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.Evaluator import Evaluator
from src.eval.utils import get_predictionFP, get_dirAndRunIdx_fromPredictionFp

from src.utils.utils import read_jsonl, convert_listOfDict_toDictOfList


def compute_avgLogProb_fromSumLogProb(prediction):
    """

    Args:
        prediction:

    Returns:

    """
    len_allChoices = torch.tensor(prediction["len_all_choices"])
    sumLogProb_ofAllChoiceIds = torch.tensor(prediction["score_of_choices"])

    allChoices_logProb = sumLogProb_ofAllChoiceIds / len_allChoices
    predicted_choice = torch.argmax(allChoices_logProb)

    prediction.update(
        {"score_of_choices": allChoices_logProb, "predicted_choice": predicted_choice}
    )

    return prediction


def compute_sumLogProb_fromAvgLogProb(prediction):
    """

    Args:
        prediction:

    Returns:

    """
    len_allChoices = torch.tensor(prediction["len_all_choices"])
    avgLogProb_ofAllChoiceIds = torch.tensor(prediction["score_of_choices"])

    allChoices_logProb = avgLogProb_ofAllChoiceIds * len_allChoices
    predicted_choice = torch.argmax(allChoices_logProb)

    prediction.update(
        {"score_of_choices": allChoices_logProb, "predicted_choice": predicted_choice}
    )

    return prediction


def identity_fn(prediction):
    """

    Args:
        prediction:

    Returns:

    """
    return prediction


def computeScore_fromCache_perFile(filepath, metrics, computeScoreFromCache_fn):
    """

    Args:
        filepath:
        metrics:
        computeScoreFromCache_fn:

    Returns:

    """
    json_predictions = read_jsonl(filepath)

    evaluator = Evaluator(metrics, None, False)

    new_predictions = []
    for prediction in json_predictions:
        new_predictions.append(computeScoreFromCache_fn(prediction))

    evaluator.add_batch(convert_listOfDict_toDictOfList(new_predictions))

    return evaluator.get_result()


def does_cachePredictionExist(
    evaluation_config, specificPrediction_dir, keys_mustMatch
):
    """

    Args:
        evaluation_config:
        specificPrediction_dir:
        keys_mustMatch:

    Returns:

    """
    evaluationRuns_fp = os.path.join(specificPrediction_dir, "evaluation_runs.json")

    canUseCached_predictionFP = False
    idxOf_predictionFP = 0
    evaluationConfigDict_runs = []
    cached_evaluationConfigDict_toUse = None

    if os.path.exists(evaluationRuns_fp):
        logger.info(f"Found cached runs {evaluationRuns_fp}")

        evaluationConfigDict_runs = json.load(open(evaluationRuns_fp, "r"))

        for idx, cached_evaluationConfigDict in enumerate(evaluationConfigDict_runs):
            if canUse_cachedEvaluationConfig(
                evaluation_config.get_dict(),
                cached_evaluationConfigDict,
                keys_mustMatch,
            ):
                canUseCached_predictionFP = True
                idxOf_predictionFP = idx
                cached_evaluationConfigDict_toUse = cached_evaluationConfigDict

        if not canUseCached_predictionFP:
            idxOf_predictionFP = len(evaluationConfigDict_runs)

    if not canUseCached_predictionFP:
        evaluationConfigDict_runs.append(evaluation_config.get_dict())
        json.dump(evaluationConfigDict_runs, open(evaluationRuns_fp, "w+"))

    prediction_fp = get_predictionFP(specificPrediction_dir, idxOf_predictionFP)

    return canUseCached_predictionFP, cached_evaluationConfigDict_toUse, prediction_fp


def canUse_cachedEvaluationConfig(
    evaluationConfig_dict, cachedEvaluationConfig_dict, keys_mustMatch
):
    """
    Check if the cached EvaluationConfig with its predictions can be used or not depending on if
    certain keys match

    Args:
        evaluationConfig_dict:
        cachedEvaluationConfig_dict:
        keys_mustMatch:

    Returns:

    """
    assert set(cachedEvaluationConfig_dict.keys()) == set(evaluationConfig_dict.keys())

    for key in keys_mustMatch:
        assert (
            key in cachedEvaluationConfig_dict.keys()
        ), f"{key} in set of keys that must match not found in config"

    canUse_cachedPredictions = True
    for key in cachedEvaluationConfig_dict.keys():
        if cachedEvaluationConfig_dict[key] != evaluationConfig_dict[key]:
            if key in keys_mustMatch:
                logger.info(
                    f"{key} required to match but doesn't and thus configs don't match "
                    f"enough to allow us to use the cached predictions"
                )
                canUse_cachedPredictions = False
            else:
                logger.info(f"{key} not matching but not required to match ")

    if not cachedEvaluationConfig_dict["did_run_finish"]:
        logger.info("Found cached evaluation run that crashed")
        canUse_cachedPredictions = False

    return canUse_cachedPredictions


def getCached_predictions(
    evaluation_config, cached_evaluationConfigDict_toUse, prediction_fp, metrics
):
    """

    Args:
        evaluation_config:
        cached_evaluationConfigDict_toUse:
        prediction_fp:
        metrics:

    Returns:

    """
    logger.info(f"Found cached prediction file {prediction_fp}")

    cached_evaluationConfig = EvaluationConfig(cached_evaluationConfigDict_toUse)

    if (
        evaluation_config.length_normalization
        == cached_evaluationConfig.length_normalization
        or "Accuracy" not in metrics
    ):
        computeScoreFromCache_fn = identity_fn
        logger.info(f"Length Normalization same as cached or Accuracy not in metrics")

    # Compute the results with a different scoring function based on the cache model scores
    else:
        if (
            evaluation_config.length_normalization
            and not cached_evaluationConfig.length_normalization
        ):
            computeScoreFromCache_fn = compute_avgLogProb_fromSumLogProb
            logger.info(f"Computing length normalized scores from cache")

        if (
            not evaluation_config.length_normalization
            and cached_evaluationConfig.length_normalization
        ):
            computeScoreFromCache_fn = compute_sumLogProb_fromAvgLogProb
            logger.info(f"Computing non length normalized scores from cache")

    return computeScore_fromCache_perFile(
        prediction_fp, metrics, computeScoreFromCache_fn
    )
