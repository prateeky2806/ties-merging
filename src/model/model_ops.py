import math
import torch
import logging
from collections.abc import Iterable

logger = logging.getLogger("root")


def reduce_modelParameters(model_parameters, reduceValue_fn, reduceCheckpoint_fn):
    """
    Reduce checkpoint into a single value

    Args:
        model_parameters:
        reduceValue_fn: Function to reduce parameter block into a single value.
        reduceCheckpoint_fn: Function to reduce values from each parameter block into a single value.

    Returns:

    """
    newModel_parameters = {}
    for parameter_name, parameter_values in model_parameters.items():
        newModel_parameters[parameter_name] = reduceValue_fn(parameter_values)

    return reduceCheckpoint_fn(list(newModel_parameters.values()))


def reduceAll_modelParameters(allModels_parameters, reduce_fn):
    """
    Reduce a list of checkpoints into a single checkpoint

    Args:
        allModels_parameters: List of dictionaries
        reduce_fn: Takes a tensor where the first dimension iterates over checkpoints
    Returns:
        Model: dictionary
    """
    # Returns list of list of parameters where the outer list is the parameter names,
    # and inner list is the models.
    all_parameterValues = zip(*list(map(lambda x: x.values(), allModels_parameters)))

    # All models must have the same parameters
    all_parameterNames = allModels_parameters[0].keys()

    newModel_parameters = {}
    for parameter_name, parameter_values in zip(
        *[all_parameterNames, all_parameterValues]
    ):
        newModel_parameters[parameter_name] = reduce_fn(
            torch.stack(list(parameter_values), dim=0)
        )

    return newModel_parameters


def pairwiseMap_modelParameters(modelOne_parameters, modelTwo_parameters, map_fn):
    """

    Args:
        modelOne_parameters:
        modelTwo_parameters:
        map_fn:

    Returns:

    """
    # All models must have the same parameters
    all_parameterNames = modelOne_parameters.keys()

    newModel_parameters = {}
    for parameter_name in all_parameterNames:
        newModel_parameters[parameter_name] = map_fn(
            modelOne_parameters[parameter_name], modelTwo_parameters[parameter_name]
        )

    return newModel_parameters


def map_modelParameters(model_parameters, map_fn):
    """

    Args:
        model_parameters:
        map_fn:

    Returns:

    """
    newModel_parameters = {}
    for parameter_name, parameter_value in model_parameters.items():
        newModel_parameters[parameter_name] = map_fn(parameter_value)
    return newModel_parameters


def convex_combination(modelOne_parameters, modelTwo_parameters, alpha, map_fn):
    """

    Args:
        modelOne_parameters:
        modelTwo_parameters:
        map_fn:

    Returns:

    """
    # All models must have the same parameters
    all_parameterNames = modelOne_parameters.keys()

    newModel_parameters = {}
    for parameter_name in all_parameterNames:
        newModel_parameters[parameter_name] = map_fn(
            modelOne_parameters[parameter_name],
            modelTwo_parameters[parameter_name],
            alpha,
        )

    return newModel_parameters


def add(modelOne_parameters, modelTwo_parameters):
    """
    Add the parameters of two models.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    add_fn = lambda x, y: x + y
    added_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, add_fn
    )
    return added_model


def subtract(modelOne_parameters, modelTwo_parameters):
    """
    Subtract the parameters of modelTwo from the parameters of modelOne.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    subtract_fn = lambda x, y: x - y
    subtracted_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, subtract_fn
    )
    return subtracted_model


def scale(model_parameters, scaler):
    """
    Multiply model parameters by scaler.

    Args:
        model_parameters:
        scaler:

    Returns:

    """
    scale_fn = lambda x: x * scaler
    scaled_model = map_modelParameters(model_parameters, scale_fn)
    return scaled_model


def random_parameters(model_parameters, model_norm):
    """
    Create model parameters to have the same shape as model parameters
    but are randomly initialized with specified norm.

    Args:
        model_parameters:
        model_norm:

    Returns:

    """
    random_fn = lambda x: torch.randn(x.size())
    random_model = map_modelParameters(model_parameters, random_fn)
    scaler = model_norm / norm(random_model)
    return scale(random_model, scaler)


def scale_andSum(allModels_parameters, model_lambda):
    """
    Scale up a list of model parameters, and then sum them.

    Args:
        allModels_parameters: List of dictionaries that represent model parameters.
        model_lambda:

    Returns:

    """
    if isinstance(model_lambda, int) or isinstance(model_lambda, float):
        logger.info(f"Scaling up all models using a single lambda {model_lambda}.")
        sum_fn = lambda parameters: torch.sum(parameters * model_lambda, dim=0)
    elif isinstance(model_lambda, Iterable):
        logger.info(f"Scaling up each model with a specific lambda {model_lambda}.")
        sum_fn = lambda parameters: torch.sum(
            parameters * torch.tensor(model_lambda).unsqueeze(-1), dim=0
        )
    else:
        raise ValueError(
            f"model_lambda must be either int, float, or Iterable. Got {type(model_lambda)}"
        )
    summed_model = reduceAll_modelParameters(allModels_parameters, sum_fn)
    return summed_model


def norm(model_parameters):
    """
    Find the L2 norm of the model parameters.

    Args:
        model_parameters:

    Returns:

    """
    square_fn = lambda x: x * x

    squared_model = map_modelParameters(model_parameters, square_fn)
    sumOfSquared_model = reduce_modelParameters(
        squared_model, lambda x: torch.sum(x), lambda x: sum(x)
    )
    return math.sqrt(sumOfSquared_model)


def dot_product(modelOne_parameters, modelTwo_parameters):
    """
    Find the dot product between the parameters of two models.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    multiply_fn = lambda x, y: x * y

    multiplied_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, multiply_fn
    )
    sumOfMultiplied_model = reduce_modelParameters(
        multiplied_model, lambda x: torch.sum(x), lambda x: sum(x)
    )
    return sumOfMultiplied_model


def cosine_sim(modelOne_parameters, modelTwo_parameters):
    """
    Find the cosine similarity between parameters of two models.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    multiply_fn = lambda x, y: x * y

    multiplied_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, multiply_fn
    )
    sumOfMultiplied_model = reduce_modelParameters(
        multiplied_model, lambda x: torch.sum(x), lambda x: sum(x)
    )
    cosine_sim = sumOfMultiplied_model.item() / (
        norm(modelOne_parameters) * norm(modelTwo_parameters)
    )

    return cosine_sim


def project(modelOne_parameters, modelTwo_parameters):
    """
    Project modelOne (a) onto modelTwo (b)

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """

    b_dot_b = dot_product(modelTwo_parameters, modelTwo_parameters)
    a_dot_b = dot_product(modelOne_parameters, modelTwo_parameters)

    scaler = a_dot_b / b_dot_b
    multiply_fn = lambda x: x * scaler

    return map_modelParameters(modelTwo_parameters, multiply_fn)


def inverse(model_parameters):
    """
    Find the inverse of the model parameters.

    Args:
        model_parameters:

    Returns:

    """
    inverse_fn = lambda x: torch.linalg.inv(x)
    inverse_model = {}
    for parameter_name, parameter_value in model_parameters.items():
        idx = (torch.diag(parameter_value) == 0).nonzero()
        if len(idx) != 0:
            parameter_value[idx.squeeze(), idx.squeeze()] = 1e-6
        inverse_model[parameter_name] = inverse_fn(parameter_value)
    return inverse_model


def random_parameters(model_parameters, model_norm):
    """
    Create model parameters to have the same shape as model parameters
    but are randomly initialized with specified norm.

    Args:
        model_parameters:
        model_norm:

    Returns:

    """
    random_fn = lambda x: torch.randn(x.size())
    random_model = map_modelParameters(model_parameters, random_fn)
    scaler = model_norm / norm(random_model)
    return scale(random_model, scaler)


def matmul(modelOne_parameters, modelTwo_parameters):
    """
    Matrix multiply the parameters.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    matmul_fn = lambda x, y: torch.matmul(x, y)
    matmul_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, matmul_fn
    )
    return matmul_model
