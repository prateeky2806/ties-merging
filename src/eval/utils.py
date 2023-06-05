import copy
import torch
import os


def prepare_batchOfEvalInfo(batch):
    batchOf_evalInfo = copy.deepcopy(batch)

    for (key, value) in batch.items():
        # Remove ids and mask since no longer needed
        if ("ids" in key) or ("mask" in key):
            del batchOf_evalInfo[key]
        else:
            # Convert tensors to list
            if torch.is_tensor(batchOf_evalInfo[key]):
                batchOf_evalInfo[key] = value.cpu().numpy().tolist()

    return batchOf_evalInfo


def getAndMake_specificPredictionDir(prediction_dir, split, dataset, template_idx):
    """

    Args:
        prediction_dir:
        dataset:
        template_idx:

    Returns:

    """

    prediction_name = f"{dataset}_template_{template_idx}"

    specificPrediction_dir = os.path.join(prediction_dir, split, prediction_name)
    if not os.path.exists(specificPrediction_dir):
        os.makedirs(specificPrediction_dir)

    return specificPrediction_dir


def get_predictionFP(specificPrediction_dir, idx):
    return os.path.join(specificPrediction_dir, f"run_{idx}.txt")


def get_dirAndRunIdx_fromPredictionFp(prediction_fp):
    """

    Args:
        prediction_fp:

    Returns:

    """
    directory = os.path.dirname(prediction_fp)
    run_filename = os.path.basename(prediction_fp)
    run_idx = int(run_filename.replace("run_", "").replace(".txt", ""))
    return directory, run_idx
