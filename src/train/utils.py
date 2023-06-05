import logging
import os
import torch
import re

from transformers import Adafactor
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

import torch.optim as optim

from src.model.load_model import get_modelParameters

logger = logging.getLogger("root")


def construct_optimizer(
    model, trainableParameter_regex, optimizer_name, learning_rate, weight_decay
):
    """


    Args:
        model:
        trainableParameter_regex:
        optimizer_name:
        learning_rate:
        weight_decay:

    Returns:

    """
    parameters, names = get_modelParameters(
        model, trainableParameter_regex, return_names=True
    )
    logger.info(f"Training the parameters: {names}")

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)

    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay, eps=1e-8
        )

    elif optimizer_name.lower() == "adafactor":
        optimizer = Adafactor(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            decay_rate=0,
            relative_step=False,
        )

    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented yet ")

    return optimizer


def construct_scheduler(optimizer, scheduler_name, num_batches, warmup_ratio):

    num_warmup_steps = num_batches * warmup_ratio

    if scheduler_name == "polynomial_decay_with_warmup":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_batches
        )

    elif scheduler_name == "exponential_decay":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer)

    elif scheduler_name == "linear_decay_with_warmup":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_batches)

    elif scheduler_name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batches)

    else:
        raise ValueError(f"scheduler {scheduler_name} not implemented")


def freeze_model_params(model, trainableParameter_regex):
    for parameter_value in model.parameters():
        parameter_value.requires_grad = False

    for parameter_name, parameter_value in model.named_parameters():
        if re.fullmatch(trainableParameter_regex, parameter_name):
            parameter_value.requires_grad = True
    return model


def load_checkpointToResumeFrom(
    resume_checkpoint_filepath, model, optimizer, scheduler
):
    """

    Args:
        resume_checkpoint_filepath:
        model:
        optimizer:
        scheduler:

    Returns:

    """
    resumeCheckpoint_dict = torch.load(resume_checkpoint_filepath)

    optimizer.load_state_dict(resumeCheckpoint_dict["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(resumeCheckpoint_dict["scheduler"])
    batchIdx_toResumeFrom = resumeCheckpoint_dict["num_batches"]

    model_fp = os.path.join(
        os.path.dirname(resume_checkpoint_filepath),
        "checkpoints",
        f"checkpoint_{batchIdx_toResumeFrom}.pt",
    )
    model.load_state_dict(torch.load(model_fp))

    current_bestScore = resumeCheckpoint_dict["current_best_score"]

    return model, optimizer, scheduler, batchIdx_toResumeFrom, current_bestScore
