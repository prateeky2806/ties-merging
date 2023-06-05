import logging
import os
import torch
import copy
import re
from collections import OrderedDict

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.data.dataset_mixtures import get_datasetMixture

from src.utils.utils import parse_modelName, map_forDictionaries

from src.model.checkpoint_filepaths import (
    generate_checkpointFilepath,
    generate_MixtureCheckpointFilepath,
)
from src.model.T5Wrapper import T5Wrapper
from src.model.merge import merge_checkpoints
from src.model.ia3 import modify_withIA3
from src.model.merge import get_taskVectors, get_orthogonalVectors
from src.model.utils import get_parameterCount

logger = logging.getLogger("root")


def construct_model(
    pretrainedModel_name,
    peft_method,
    max_sequenceLength,
    device,
    model_config=None,
):
    """


    Args:
        model_name:
        peft_method:
        max_seq_len:
        device

    Returns:
        model:
        transformer:
        tokenizer:
    """
    logger.info(f"Loading pretrained model and tokenizer for {pretrainedModel_name}")
    transformer = AutoModelForSeq2SeqLM.from_pretrained(pretrainedModel_name)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrainedModel_name, model_max_length=max_sequenceLength
    )

    if peft_method is not None:
        if "ia3" in peft_method:
            transformer, trainableParameter_regex = modify_withIA3(
                transformer, model_config
            )
        else:
            raise ValueError(f"peft_method {peft_method} not recognized")
    else:
        trainableParameter_regex = ".*"

    model = T5Wrapper(transformer, tokenizer).to(device)

    return model, tokenizer, trainableParameter_regex


def load_stateDict(model_filepath):
    """
    Load model from filepath if it exists.

    Args:
        model:
        model_filepath:

    Returns:

    """
    if os.path.exists(model_filepath):
        logger.info(f"Using {model_filepath} model found")
        return torch.load(model_filepath)
    else:
        assert "Filepath not found"


def get_modelParameters(model, trainableParameter_regex, return_names=False):
    """
    Gets the model parameters and put on CPU

    Args:
        state_dict:
        parameter_names:

    Returns:

    """
    model_parameters = []
    param_names = []
    for parameter_name, parameter_value in model.named_parameters():
        if re.fullmatch(trainableParameter_regex, parameter_name):
            model_parameters.append(parameter_value)
            param_names.append(parameter_name)
    if return_names:
        return model_parameters, param_names
    return model_parameters


def get_modelParameters_stateDict(state_dict, trainableParameter_regex, device=None):
    """
    Gets the model parameters and put on CPU

    Args:
        state_dict:
        parameter_names:

    Returns:

    """
    model_parameters = {}
    for parameter_name, parameter_value in state_dict.items():
        if (
            re.fullmatch(trainableParameter_regex, parameter_name)
            or trainableParameter_regex == ""
        ):
            model_parameters[parameter_name] = (
                parameter_value.cpu() if device is None else parameter_value.to(device)
            )
    return model_parameters


def load_checkpoints(
    modelCheckpoints_fp, trainableParameter_regex, loaded_checkpoints, device=None
):
    """
    Load all the checkpoints that are not already loaded

    Args:
        modelCheckpoints_fp:
        trainableParameter_regex:
        loaded_checkpoints:

    Returns:

    """

    for checkpoint_fp in modelCheckpoints_fp.values():
        if checkpoint_fp not in loaded_checkpoints:
            loaded_checkpoints[checkpoint_fp] = get_modelParameters_stateDict(
                load_stateDict(checkpoint_fp), trainableParameter_regex, device
            )

    return loaded_checkpoints


def check_parameterNamesMatch(checkpoints):
    """
    Check that the parameter names are the same for all checkpoints

    Args:
        checkpoints:

    Returns:

    """
    checkpoint_filepaths = list(checkpoints.keys())
    parameter_names = set(checkpoints[checkpoint_filepaths[0]].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint_fp in checkpoint_filepaths[1:]:
            current_parameterNames = set(checkpoints[checkpoint_fp].keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def loadCheckpoint_intoModel(checkpoint, model):
    """_summary_

    Args:
        checkpoint (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    modelParameters_names = set(checkpoint.keys())
    modelStateDict_keys = set(model.state_dict().keys())

    assert modelParameters_names.issubset(modelStateDict_keys)
    # The encoder and decoder embedding tokens are always tied to the shared weight and so should never be a paremeter
    # assert set(
    #     [
    #         "transformer.decoder.embed_tokens.weight",
    #         "transformer.encoder.embed_tokens.weight",
    #     ]
    # ).issubset(modelStateDict_keys.difference(modelParameters_names))

    # Must tie the encoder and decoder embeddings to the shared weight if the shared weight is a parameter.
    if "transformer.shared.weight" in checkpoint:
        checkpoint["transformer.decoder.embed_tokens.weight"] = checkpoint[
            "transformer.shared.weight"
        ]
        checkpoint["transformer.encoder.embed_tokens.weight"] = checkpoint[
            "transformer.shared.weight"
        ]

    update_stats = model.load_state_dict(checkpoint, strict=False)
    logger.info(f"Missing keys: {update_stats.missing_keys}")
    logger.info(f"Unexpected keys: {update_stats.unexpected_keys}")

    if "transformer.shared.weight" in checkpoint:
        del checkpoint["transformer.decoder.embed_tokens.weight"]
        del checkpoint["transformer.encoder.embed_tokens.weight"]

    return model


def load_model(model_config, cached_models, device):
    """

    Args:
        model_config:
        cached_models:
        device:

    Returns:

    """
    # Cannot both load a checkpoint directly and merge checkpoints,
    # since loading a checkpoint directly means loading a checkpoint from disk.
    assert (model_config.checkpoint_to_directly_load_model is None) or (
        model_config.function_to_merge_checkpoints is None
    )

    if "model" not in cached_models:
        model, tokenizer, peftParameter_regex = construct_model(
            model_config.pretrained_model,
            model_config.peft_method,
            model_config.max_seq_len,
            device=device,
            model_config=model_config,
        )

        cached_models["model"] = model
        cached_models["tokenizer"] = tokenizer
        cached_models["trainable_parameter_regex"] = peftParameter_regex
    else:
        model = cached_models["model"]
        tokenizer = cached_models["tokenizer"]
        peftParameter_regex = cached_models["trainable_parameter_regex"]

    if model_config.model_checkpoints is not None:
        if "loaded_checkpoints" in cached_models:
            loaded_checkpoints = cached_models["loaded_checkpoints"]
        else:
            loaded_checkpoints = {}
        loaded_checkpoints = load_checkpoints(
            model_config.model_checkpoints,
            peftParameter_regex,
            loaded_checkpoints,
        )
        check_parameterNamesMatch(loaded_checkpoints)
        cached_models["loaded_checkpoints"] = loaded_checkpoints

        if "pretrained_checkpoint" not in cached_models:
            checkpoint_fps_list = list(model_config.model_checkpoints.values())
            if "fewshot" in checkpoint_fps_list[0]:
                pretrained_folder = "/".join(
                    x for x in checkpoint_fps_list[0].split("/")[:-3]
                )
            elif "fullshot" in checkpoint_fps_list[0]:
                pretrained_folder = "/".join(
                    x for x in checkpoint_fps_list[0].split("/")[:-2]
                )
            else:
                raise ValueError("Invalid checkpoint filepath")

            pretrained_fp = os.path.join(pretrained_folder, "pretrained.pt")
            assert os.path.exists(
                pretrained_fp
            ), f"Pretrained checkpoint does not exist at {pretrained_fp}"
            logger.info(f"\tLoading/Using Pretrained checkpoint: {pretrained_fp}")
            cached_models["pretrained_checkpoint"] = torch.load(pretrained_fp)

    if model_config.checkpoint_to_directly_load_model is not None:
        checkpoint_path = generate_checkpointFilepath(
            model_config.pretrained_model,
            model_config.peft_method,
            model_config.checkpoint_to_directly_load_model,
        )
        state_dict = load_stateDict(checkpoint_path)
        logger.info(f"\tLoaded Checkpoint from {checkpoint_path}")
        model = loadCheckpoint_intoModel(state_dict, model)

    if model_config.function_to_merge_checkpoints is not None:
        assert (
            model_config.model_checkpoints is not None
        ), "Checkpoints must be specified if we want to merge them."
        merged_checkpoint = merge_checkpoints(
            model_config,
            cached_models["loaded_checkpoints"],
            cached_models["pretrained_checkpoint"],
        )
        model = loadCheckpoint_intoModel(merged_checkpoint, model)

    return model, tokenizer, peftParameter_regex, cached_models


def get_pretrained_model(model_config, device):
    model, tokenizer, trainableParameter_regex = construct_model(
        model_config.pretrained_model,
        model_config.peft_method,
        model_config.max_seq_len,
        device=device,
        model_config=model_config,
    )
    return model, tokenizer, trainableParameter_regex


def concatenate_ia3_checkpoint_list(checkpoints, device=None):
    concatenated_checkpoint = {}
    param_keys = checkpoints[0].keys()
    for pname in param_keys:
        params = []
        for checkpoint_params in checkpoints:
            params.append(checkpoint_params[pname].unsqueeze(0))

        cat_params = torch.cat(params, axis=0)
        concatenated_checkpoint[pname] = (
            cat_params.to(device) if device is not None else cat_params
        )
    return concatenated_checkpoint


def get_Checkpoint_and_Vectors(
    checkpoint_fps, peft_param_regex, merge_func, device=None
):
    assert checkpoint_fps is not None

    loaded_checkpoints = OrderedDict()
    loaded_checkpoints = load_checkpoints(
        checkpoint_fps,
        peft_param_regex,
        loaded_checkpoints,
        device=device,
    )
    check_parameterNamesMatch(loaded_checkpoints)

    checkpoint_fps_list = list(checkpoint_fps.values())
    if "fewshot" in checkpoint_fps_list[0]:
        pretrained_folder = "/".join(x for x in checkpoint_fps_list[0].split("/")[:-3])
    elif "fullshot" in checkpoint_fps_list[0]:
        pretrained_folder = "/".join(x for x in checkpoint_fps_list[0].split("/")[:-2])
    else:
        raise ValueError("Invalid checkpoint filepath")

    pretrained_fp = os.path.join(pretrained_folder, "pretrained.pt")
    logger.info(f"\tLoading Pretrained checkpoint: {pretrained_fp}")

    pretrained_checkpoint = torch.load(pretrained_fp)

    if merge_func == "ia3":
        return_vectors = loaded_checkpoints
    elif merge_func == "task_vectors":
        return_vectors = get_taskVectors(loaded_checkpoints, pretrained_checkpoint)
    elif merge_func == "orthogonal_task_vectors":
        return_vectors = get_orthogonalVectors(loaded_checkpoints)
    else:
        raise NotImplementedError(f"Merge function {merge_func} not implemented.")

    return return_vectors, loaded_checkpoints, pretrained_checkpoint


def get_Checkpoint_and_load_multiIA3_Vectors(
    model_config, dataset_mixture_to_merge, merge_func, device=None
):

    modelCheckpoints_fps = generate_MixtureCheckpointFilepath(
        model_config.pretrained_model,
        model_config.peft_method,
        dataset_mixture_to_merge,
    )
    (
        vectors,
        loaded_checkpoints,
        pretrained_checkpoint,
    ) = get_Checkpoint_and_Vectors(modelCheckpoints_fps, "", merge_func)

    model_config.num_vectors = len(vectors)
    model, tokenizer, peftParameter_regex = construct_model(
        model_config.pretrained_model,
        model_config.peft_method,
        model_config.max_seq_len,
        device=device,
        model_config=model_config,
    )

    if "task_vectors" not in merge_func:
        ia3_concat_list = list(vectors.values())
    else:
        ia3_concat_list = [pretrained_checkpoint] + list(vectors.values())

    concat_ia3_Vector = concatenate_ia3_checkpoint_list(ia3_concat_list, device)
    model = loadCheckpoint_intoModel(concat_ia3_Vector, model)

    logger.info(
        f"{model_config.peft_method.upper()} Parameter count: {get_parameterCount(model, peftParameter_regex) * len(vectors)}"
    )

    return (
        model,
        tokenizer,
        peftParameter_regex,
        vectors,
        pretrained_checkpoint,
        loaded_checkpoints,
    )
