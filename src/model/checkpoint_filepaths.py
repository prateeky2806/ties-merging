import os
from src.utils.utils import parse_modelName

from src.data.dataset_mixtures import get_datasetMixture
from src.data.dataset_readers import DATASET_CLASSES
from src.data.dataset_mixtures import DATASET_MIXTURES

model_map = {"bigscience-T0_3B": "t03b"}
method_folder_map = {}


def generate_checkpointFilepath(
    pretrained_model,
    peft_method,
    checkpoint_to_directly_load_model,
    basedir="",
):
    if "+" in checkpoint_to_directly_load_model:
        if "checkpoint" in checkpoint_to_directly_load_model:
            dataset_code, _, batch_idx = checkpoint_to_directly_load_model.split("+")
            checkpoint_path = os.path.join(
                basedir,
                "fullshot",
                f"{peft_method}",
                f"{dataset_code}",
                "checkpoints",
                f"checkpoint_{batch_idx}.pt",
            )
        else:
            if "pretrained" in checkpoint_to_directly_load_model:
                from_pretrained = checkpoint_to_directly_load_model.split("+")[-1]
                filepath = os.path.join(
                    basedir,
                    "fewshot",
                    f"{peft_method if not eval(from_pretrained) else peft_method + '_pretrained'}",
                    f"{model_map[parse_modelName(pretrained_model)]}",
                )
                checkpoint_path = os.path.join(filepath, "pretrained.pt")
            else:
                (
                    dataset,
                    seed,
                    from_pretrained,
                ) = checkpoint_to_directly_load_model.split("+")

                filepath = os.path.join(
                    basedir,
                    "fewshot",
                    f"{peft_method if not eval(from_pretrained) else peft_method + '_pretrained'}",
                    f"{model_map[parse_modelName(pretrained_model)]}",
                    f"{dataset}",
                    f"seed{seed}",
                )
                checkpoint_path = os.path.join(filepath, "finish.pt")
    elif ".pt" in checkpoint_to_directly_load_model:
        checkpoint_path = checkpoint_to_directly_load_model
    elif "*" in checkpoint_to_directly_load_model:
        dataset, other_code = checkpoint_to_directly_load_model.split('*')
        folder, factor = other_code.split("#")

        checkpoint_path = os.path.join(
            basedir,
            "fullshot",
            f"{peft_method}",
            "other_models",
            folder,
            f"{dataset}_f{factor}.pt",
        )
    else:
        checkpoint_path = os.path.join(
            basedir,
            "fullshot",
            f"{peft_method}",
            f"{checkpoint_to_directly_load_model}",
            "best.pt",
        )

    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint {checkpoint_path} does not exist!"

    return checkpoint_path


def get_fewshot_codes(dataset_codes, seed, from_pretrained):
    checkpoint_codes_to_merge = []
    for dataset_or_mixture in dataset_codes.split(","):
        if ("pretrained" in dataset_or_mixture) or (
            dataset_or_mixture in DATASET_CLASSES.keys()
        ):
            checkpoint_codes_to_merge.append(
                f"{dataset_or_mixture}+{seed}+{from_pretrained}"
            )
        elif dataset_or_mixture in DATASET_MIXTURES.keys():
            for ds in DATASET_MIXTURES[dataset_or_mixture]:
                checkpoint_codes_to_merge.append(f"{ds}+{seed}+{from_pretrained}")
    return checkpoint_codes_to_merge


def get_checkpoint_codes(dataset_codes, batch_idx):
    checkpoint_codes_to_merge = []
    for dataset_or_mixture in dataset_codes.split(","):
        checkpoint_codes_to_merge.append(
            f"{dataset_or_mixture}+checkpoint+{batch_idx}"
        )
    return checkpoint_codes_to_merge


def fullshot_codes(dataset_codes):
    checkpoint_codes_to_merge = []
    for dataset_or_mixture in dataset_codes.split(","):
        if '*' in dataset_or_mixture:
            dataset_or_mixture_code, other_code = dataset_or_mixture.split('*')
        else:
            dataset_or_mixture_code = dataset_or_mixture

        checkpoint_codes_to_merge = decode_dataset_or_mixture(dataset_or_mixture_code)

        if '*' in dataset_or_mixture:
            checkpoint_codes_to_merge = [f"{code}*{other_code}" for code in checkpoint_codes_to_merge]

    return checkpoint_codes_to_merge

def decode_dataset_or_mixture(dataset_or_mixture):
    checkpoint_codes_to_merge = []
    if dataset_or_mixture in DATASET_MIXTURES.keys():
        for ds in DATASET_MIXTURES[dataset_or_mixture]:
            checkpoint_codes_to_merge.append(ds)
    else:
        checkpoint_codes_to_merge.append(dataset_or_mixture)
    return checkpoint_codes_to_merge


def generate_MixtureCheckpointFilepath(
    pretrained_model,
    peft_method,
    dataset_mixture_to_merge,
):
    checkpoint_codes_to_merge = []
    if "+" in dataset_mixture_to_merge:
        if 'checkpoint' in dataset_mixture_to_merge:
            dataset_codes, _, batch_idx = dataset_mixture_to_merge.split("+")
            checkpoint_codes_to_merge = get_checkpoint_codes(dataset_codes, batch_idx)
        else:
            dataset_codes, seed, from_pretrained = dataset_mixture_to_merge.split("+")
            checkpoint_codes_to_merge = get_fewshot_codes(
                dataset_codes, seed, from_pretrained
            )
    else:
        checkpoint_codes_to_merge = fullshot_codes(dataset_mixture_to_merge)

    # get unique checkpoint names
    checkpoint_codes_to_merge = list(dict.fromkeys(checkpoint_codes_to_merge))

    checkpoint_fps = {}
    for dataset_or_mixture in checkpoint_codes_to_merge:
        checkpoint_path = generate_checkpointFilepath(
            pretrained_model,
            peft_method,
            dataset_or_mixture,
        )
        checkpoint_fps[dataset_or_mixture] = checkpoint_path

    return checkpoint_fps


def get_modelCheckpointFilepath(pretrained_model, checkpoint_descriptor, dataset):
    """

    Args:
        pretrained_model:
        checkpoint_descriptor:
        dataset:

    Returns:

    """
    pretrained_model = parse_modelName(pretrained_model)

    return MODEL_CHECKPOINTS[pretrained_model][checkpoint_descriptor][dataset]


def get_datasetMixtureCheckpointFilepaths(
    pretrained_model, checkpoint_descriptor, dataset_mixture
):
    """

    Args:
        pretrained_model (str):
        checkpoint_descriptor (str):
        dataset_mixture (list or str):

    Returns:
        checkpoint_fps (list):
    """

    checkpoint_fps = []
    for dataset in get_datasetMixture(dataset_mixture):
        checkpoint_fps.append(
            get_modelCheckpointFilepath(
                pretrained_model, checkpoint_descriptor, dataset
            )
        )

    return checkpoint_fps


def getDataset_fromCheckpointFilepath(filepath):
    dataset = filepath.split("/")[1]
    return dataset
