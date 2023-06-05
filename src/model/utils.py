import torch
import torch.nn as nn
import re, os

from src.data.dataset_mixtures import get_datasetMixture

def browse_all_files_in_a_directory(directory, file_extension):
    """Returns a list of all files in a directory with a given extension"""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(file_extension)]


def walk_through_all_files_in_a_directory(directory, file_extension):
    """Returns a list of all files in a directory with a given extension"""
    return [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if file.endswith(file_extension)]


def extract_number(filename):
    # Regular expression to extract a number from a string
    return int(filename.split("_")[-1].split(".")[0])


def get_parameterNames(model):
    """

    Args:
        model:

    Returns:

    """
    parameter_names = {}

    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter_names[parameter_name] = True

    return parameter_names


def get_parameterCount(model, trainableParameter_regex):
    """

    Args:
        model:

    Returns:

    """
    num_parameters = 0

    for parameter_name, parameter in model.named_parameters():
        if re.fullmatch(trainableParameter_regex, parameter_name):
            num_parameters += parameter.numel()

    return num_parameters


def get_numParameters_inCheckpoint(checkpoint):
    num_parameters = 0

    for parameter_name, parameter in checkpoint.items():
        num_parameters += parameter.numel()

    return num_parameters


def flatten_parameters(model_checkpoints):
    """

    Args:
        model_checkpoints:

    Returns:

    """
    num_parameters = 0

    for _, parameter in model_checkpoints[0].items():
        num_parameters += parameter.numel()
        device = parameter.device

    flattened_checkpoints = torch.zeros((len(model_checkpoints), num_parameters)).to(
        device
    )

    for checkpoint_idx, checkpoint in enumerate(model_checkpoints):
        current_startIdx = 0
        for _, parameter in checkpoint.items():
            end_idx = current_startIdx + parameter.numel()
            flattened_checkpoints[
                checkpoint_idx, current_startIdx:end_idx
            ] = parameter.flatten()
            current_startIdx = end_idx

    return flattened_checkpoints


def parse_peft_code(peft_code):
    if "ia3" in peft_code:
        peft_method, num_vectors = peft_code.split("+")
        num_vectors = int(num_vectors)
        return (peft_method, num_vectors)
    else:
        raise ValueError(f"peft_code {peft_code} is not valid")


def softmax_with_temperature(input, temperature, dim=-1):
    return torch.softmax(input / temperature, dim=dim)





