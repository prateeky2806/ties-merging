import copy

import torch
import logging

from src.utils.utils import map_forDictionaries

from src.model.model_ops import (
    project,
    scale_andSum,
    reduceAll_modelParameters,
    add,
    pairwiseMap_modelParameters,
    convex_combination,
    subtract,
    norm,
)

logger = logging.getLogger("root")


def average(model_lambda, loadedModels_parameters, mean=True):
    """

    Args:
        model_lambda:
        loadedModels_parameters: Dictionary mapping checkpoint filepaths to dictionary of parameters.

    Returns:

    """
    checkpoints = list(loadedModels_parameters.values())

    logger.info(
        f"Taking the average of {len(checkpoints)} checkpoints with lambda {model_lambda}"
    )

    # Divide by number of checkpoints to get the average.
    if mean:
        scaling_factor = model_lambda / len(checkpoints)
    else:
        scaling_factor = model_lambda
    average_model = scale_andSum(checkpoints, scaling_factor)

    return average_model


def get_taskVectors(task_models, pretrained_model):
    """

    Args:
        task_models:
        pretrained_model:

    Returns:

    """
    taskVector_models = map_forDictionaries(
        task_models, lambda checkpoint: subtract(checkpoint, pretrained_model)
    )
    return taskVector_models


def task_vectors(
    model_lambda, loadedModels_parameters, pretrained_parameters, nopt=False
):
    """

    Args:
        model_lambda:
        loadedModels_parameters: Dictionary mapping checkpoint filepaths to dictionary of parameters.
        pretrained_parameters: Dictionary of parameters.

    Returns:

    """
    assert pretrained_parameters is not None
    logger.info(
        f"Taking the task vectors of {len(loadedModels_parameters)} checkpoints with lambda {model_lambda}"
    )

    taskVector_models = get_taskVectors(loadedModels_parameters, pretrained_parameters)
    taskVector_models = list(taskVector_models.values())

    summed_model = scale_andSum(taskVector_models, model_lambda)

    if nopt:
        return summed_model
    else:
        final_model = add(summed_model, pretrained_parameters)
        return final_model


def get_orthogonalVectors(models):
    """
    Return models where each model is orthogonal to all other models.

    Args:
        models:

    Returns:

    """
    model_names = list(models.keys())
    model_parameters = list(models.values())
    num_models = len(model_names)

    orthogonalTaskVector_models = {model_names[0]: copy.deepcopy(model_parameters[0])}

    for model_idx in range(1, num_models):
        current_model = copy.deepcopy(model_parameters[model_idx])

        for base_model in orthogonalTaskVector_models.values():
            projection = project(current_model, base_model)
            current_model = subtract(current_model, projection)

        orthogonalTaskVector_models[model_names[model_idx]] = current_model

    return orthogonalTaskVector_models


def orthogonal_task_vectors(
    model_lambda, loadedModels_parameters, pretrained_parameters, nopt=False
):
    """

    Args:
        model_lambda:
        loadedModels_parameters: Dictionary mapping checkpoint filepaths to dictionary of parameters.
        pretrained_parameters: Dictionary of parameters.

    Returns:

    """

    assert pretrained_parameters is not None
    logger.info(
        f"Taking the orthogonal task vectors of {len(loadedModels_parameters)} checkpoints with lambda {model_lambda}"
    )

    taskVector_models = get_taskVectors(loadedModels_parameters, pretrained_parameters)
    orthogonalTaskVector_models = get_orthogonalVectors(taskVector_models)

    orthogonalTaskVector_models = list(orthogonalTaskVector_models.values())
    summed_model = scale_andSum(orthogonalTaskVector_models, model_lambda)

    if nopt:
        return summed_model
    else:
        final_model = add(summed_model, pretrained_parameters)
        return final_model


def merge_checkpoints(model_config, loadedModels_parameters, pretrained_parameters):

    if "base" in model_config.function_to_merge_checkpoints:
        if "average" in model_config.function_to_merge_checkpoints:
            checkpoint = average(
                model_config.model_lambda, loadedModels_parameters, mean=True
            )
        elif "sum" in model_config.function_to_merge_checkpoints:
            checkpoint = average(
                model_config.model_lambda, loadedModels_parameters, mean=False
            )
        else:
            raise ValueError(
                f"Merge function {model_config.function_to_merge_checkpoints} is not valid."
            )

        if "withpt" in model_config.function_to_merge_checkpoints:
            logger.info("Adding pretrained model to the checkpoint.")
            final_model = add(checkpoint, pretrained_parameters)
            return final_model
        return checkpoint

    elif ("task_vectors" in model_config.function_to_merge_checkpoints) and (
        "othogonal" not in model_config.function_to_merge_checkpoints
    ):
        return task_vectors(
            model_config.model_lambda,
            loadedModels_parameters,
            pretrained_parameters,
            nopt=True
            if "nopt" in model_config.function_to_merge_checkpoints
            else False,
        )

    elif "orthogonal_task_vectors" in model_config.function_to_merge_checkpoints:
        return orthogonal_task_vectors(
            model_config.model_lambda,
            loadedModels_parameters,
            pretrained_parameters,
            nopt=True
            if "nopt" in model_config.function_to_merge_checkpoints
            else False,
        )

    elif model_config.function_to_merge_checkpoints == "convex":
        assert (
            len(loadedModels_parameters) == 2
        ), f"Robust merge only works with exactly 2 checkpoint. Got {len(loadedModels_parameters)} checkpoints."
        checkpoint1 = list(loadedModels_parameters.values())[0]
        checkpoint2 = list(loadedModels_parameters.values())[1]
        map_func = lambda c1, c2, lam: (1 - lam) * c1 + lam * c2
        return convex_combination(
            checkpoint1, checkpoint2, model_config.model_lambda, map_func
        )


    elif model_config.function_to_merge_checkpoints == "robust":
        assert (
            len(loadedModels_parameters) == 1
        ), f"Robust merge only works with 1 checkpoint. Got {len(loadedModels_parameters)} checkpoints."
        checkpoint = list(loadedModels_parameters.values())[0]
        map_func = lambda x, y, l: (1 - l) * x + l * y
        return convex_combination(
            pretrained_parameters, checkpoint, model_config.model_lambda, map_func
        )

    else:
        raise ValueError(
            f"Merge function {model_config.function_to_merge_checkpoints} is not valid."
        )
