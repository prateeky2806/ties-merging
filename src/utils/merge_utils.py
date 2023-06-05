import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())


import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
from src.utils.analysis_utils import *
from transformers import AutoModelForSeq2SeqLM
import torch.nn.functional as F

logger = logging.getLogger("root")

BASIC_INFORMATION = {
    "bigscience/T0_3B": {
        "base_model": "bigscience/T0_3B",
        "dump_dir": f"",
        "load_dir": f"",
    },
    "t5-base": {
        "base_model": "t5-base",
        "dump_dir": f"",
        "load_dir": f"",
    },
    "t5-large": {
        "base_model": "t5-large",
        "dump_dir": f"",
        "load_dir": f"",
    },
}


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


def greater_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() > factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


def less_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() < factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def bottomk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)

    # Create a mask tensor with True for the bottom k elements in each row
    mask = M.abs() <= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def topk_mask_preserve_normfrac(T, normfrac=0.9, return_mask=False):
    row_norms = torch.norm(T, p=2, dim=1, keepdim=True)

    # Calculate the proportion of each element's contribution to its row's norm
    proportion = T.abs() ** 2 / row_norms ** 2

    # Sort the proportions and their indices in descending order
    sorted_proportions, sorted_indices = torch.sort(proportion, dim=1, descending=True)

    # Calculate the cumulative sum of proportions
    cumsum_proportions = torch.cumsum(sorted_proportions, dim=1)

    # Find the indices where cumulative sum >= normfrac
    normfrac_mask = cumsum_proportions >= normfrac
    normfrac_indices = torch.argmax(normfrac_mask.float(), dim=1)

    # Create a range tensor to compare with normfrac_indices
    range_tensor = torch.arange(T.size(1)).unsqueeze(0).expand(T.size(0), -1)

    # Create a mask based on the normfrac_indices
    mask = range_tensor <= normfrac_indices.unsqueeze(1)

    # Initialize final_indices with a value that is out of bounds
    final_indices = torch.full_like(sorted_indices, T.size(1) - 1)

    # Use the mask to get the final indices
    final_indices[mask] = sorted_indices[mask]

    # Initialize the mask with zeros
    M = torch.zeros_like(T, dtype=torch.bool)

    # Use the final indices to update the final mask M
    M.scatter_(1, final_indices, True)

    if return_mask:
        return (T * M), M.float().mean(dim=1), M
    else:
        return (T * M), M.float().mean(dim=1)


def sign_agreement_ratio(M):
    positive_count = (M > 0).float().sum(dim=0)
    negative_count = (M < 0).float().sum(dim=0)

    non_zero_values = positive_count + negative_count

    sar = torch.where(
        non_zero_values != 0,
        torch.abs(positive_count - negative_count) / non_zero_values,
        torch.ones_like(non_zero_values),
    )

    return sar


def replace_noise_and_constant(tensor, mask, replace_factor, sign_tensor):
    tensor[mask] = 0

    if replace_factor != 0:
        tensor[~mask] = replace_factor * tensor.std()
        tensor *= sign_tensor

    return tensor


def plot_topk_norms(flat_checkpoints, task_names, topks):
    check_norms = flat_checkpoints.norm(dim=1)

    all_topk_norm = []
    for topk in topks:
        print(topk)
        topk_vector, *_ = topk_values_mask(flat_checkpoints, K=topk)
        topk_norms = topk_vector.norm(dim=1)
        all_topk_norm.append(topk_norms)
    all_topk_norm = torch.vstack(all_topk_norm)
    task_topk_norms = all_topk_norm.T
    normalized_task_topk_norms = task_topk_norms / check_norms.unsqueeze(1)

    plot_rows(
        np.array(topks),
        np.array(normalized_task_topk_norms),
        labels=[f"{task_names[i]}" for i in range(len(flat_checkpoints))],
        title="Norm of top-k% parameters over total norm for each task",
        x_label="top-k%",
        y_label="Norm@k",
    )


def plot_rows(X, Y, labels=None, title=None, x_label=None, y_label=None):
    if X.shape[0] != Y.shape[1]:
        raise ValueError(
            "Length of vector X must match the number of columns in matrix Y."
        )

    if labels is not None and len(labels) != Y.shape[0]:
        raise ValueError("Number of labels must match the number of rows in matrix Y.")

    for i in range(Y.shape[0]):
        plt.plot(X, Y[i], label=labels[i] if labels is not None else None)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    if labels is not None:
        plt.legend()

    plt.show()


def plot_row_histograms(M, task_names, bins=None):
    n, d = M.shape
    num_cols = min(
        n, 4
    )  # Adjust this value to change the number of columns in the subplot grid
    num_rows = (n + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

    for i, row in enumerate(M):
        ax = (
            axes[i // num_cols, i % num_cols]
            if num_rows > 1 or num_cols > 1
            else axes[i]
        )
        ax.hist(row.numpy(), bins=bins)
        ax.set_title(f"{task_names[i]}")
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")

    # Remove extra subplots if any
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])

    plt.tight_layout()
    plt.show()


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def normfrac_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return torch.sign(Tensor[norm_fracs.argmax(dim=0), torch.arange(Tensor.shape[1])])


def normmass_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return (Tensor.sign() * norm_fracs.abs()).sum(dim=0).sign()


def resolve_sign(Tensor, resolve_method):
    if resolve_method == "mass":
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
    elif resolve_method == "normfrac":
        sign_to_mult = normfrac_based_sign(Tensor)
    elif resolve_method == "normmass":
        sign_to_mult = normmass_based_sign(Tensor)
    else:
        raise ValueError(f"Sign resolve method {resolve_method} is not defined.")
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def aggregate(T, agg_type, final_signs, dim=0):
    if agg_type == "mean":
        result = torch.mean(T, dim=dim)
    elif agg_type == "sum":
        result = torch.sum(T, dim=dim)
    elif agg_type == "median":
        result = torch.median(T, dim=dim)[0]
    elif agg_type == "magnitude":
        max_indices = T.abs().argmax(dim=0)
        result = T[max_indices, torch.arange(T.shape[1])]
    else:
        raise ValueError("Invalid agg_type: %s" % agg_type)

    if final_signs is not None:
        # print(final_signs)
        result = result.abs() * final_signs

    return result


def basic_merging(merge_func, flat_checks, sd_checks, remove_keys):
    """ "Basic aggregation of the delta checks"""
    all_checks = flat_checks.clone()
    merged_check = aggregate(all_checks, merge_func, final_signs=None)

    final_sd = vector_to_state_dict(merged_check, sd_checks[0], remove_keys=remove_keys)
    return final_sd


def tv_merging(tv_flat_checks):
    """Merging by creating and scaling Task Vectors"""
    all_checks = tv_flat_checks.clone()
    tv_merged_check = aggregate(all_checks, "sum", final_signs=None)
    return tv_merged_check


def merge_methods(
    reset_type,
    flat_task_checks,
    reset_thresh=None,
    resolve_method=None,
    merge_func="",
):
    all_checks = flat_task_checks.clone()

    if "nf" in reset_type and reset_thresh != "none":
        logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = topk_mask_preserve_normfrac(
            all_checks, reset_thresh, return_mask=False
        )
    elif "topk" in reset_type and reset_thresh != "none":
        logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = topk_values_mask(
            all_checks, K=reset_thresh, return_mask=False
        )
    elif "std" in reset_type and reset_thresh != "none":
        logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = greater_than_std_mask(
            all_checks, reset_thresh, return_mask=False
        )
    else:
        logger.info("Not removing NOISE")
        updated_checks = all_checks

    if resolve_method != "none":
        logger.info(f"RESOLVING SIGN: {resolve_method}")
        final_signs = resolve_sign(updated_checks, resolve_method)
        assert final_signs is not None
    else:
        logger.info("Not RESOLVING SIGN")
        final_signs = None

    if "dis" in merge_func:
        logger.info(f"Disjoint AGGREGATION: {merge_func}")
        merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    else:
        logger.info(f"Basic AGGREGATION: {merge_func}")
        merged_tv = aggregate(updated_checks, merge_func, final_signs)

    return merged_tv


def merge_with_oracle_sign(
    final_signs,
    flat_task_checks,
    reset_thresh,
    merge_func,
):
    all_checks = flat_task_checks.clone()
    if reset_thresh != "none":
        logger.info(f"Pruning: {reset_thresh}")
        updated_checks, *_ = topk_values_mask(
            all_checks, K=reset_thresh, return_mask=False
        )
    else:
        logger.info("Not removing NOISE")
        updated_checks = all_checks
    # updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    return merged_tv


def load_ia3(merging_tasks, model_key):
    dump_dir = BASIC_INFORMATION[model_key]["dump_dir"]
    base_model = BASIC_INFORMATION[model_key]["base_model"]
    load_dir = BASIC_INFORMATION[model_key]["load_dir"]

    # load the finetune and pretrained checkpoints
    filepaths = [os.path.join(load_dir, task, "best.pt") for task in merging_tasks]
    ft_checks = [torch.load(fp) for fp in filepaths]
    ptm_check = {pn: torch.ones_like(pv) for pn, pv in ft_checks[0].items()}

    # check if all checkpoints have the same paramters.
    check_parameterNamesMatch(ft_checks + [ptm_check])
    remove_keys = []
    reshape_keys = []
    for key in remove_keys + reshape_keys:
        ptm_check[key] = ptm_check[key][: len(ft_checks[0][key]), :]

    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm

    # check if the vectorized state dicts can be converted back to the original state dicts.
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ft_checks[0], remove_keys), ptm_check
    )
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    )
    assert all(
        [
            check_state_dicts_equal(
                vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )

    logger.info(
        f"Pretrained Model: {base_model}\tMean:{flat_ptm.mean():.6f}\tSTD: {flat_ptm.std():.6f}"
    )
    logger.info(f"Merging tasks: {merging_tasks}")
    logger.info(
        f"Finetune checkpoint \tMean:{flat_ft.mean(dim=1)}\tSTD: {flat_ft.std(dim=1)}"
    )
    logger.info(
        f"Task Vector Finetune checkpoint\tMean:{tv_flat_checks.mean(dim=1)}\tSTD: {tv_flat_checks.std(dim=1)}"
    )

    return (
        tv_flat_checks,
        flat_ptm,
        flat_ft,
        ft_checks,
        ptm_check,
        merging_tasks,
        dump_dir,
        remove_keys,
        reshape_keys,
        base_model,
        load_dir,
        filepaths,
    )


def load_t5(merging_tasks, model_key, pretrained_state_dict):

    dump_dir = BASIC_INFORMATION[model_key]["dump_dir"]
    base_model = BASIC_INFORMATION[model_key]["base_model"]
    load_dir = BASIC_INFORMATION[model_key]["load_dir"]

    # load the finetune and pretrained checkpoints
    filepaths = [os.path.join(load_dir, task, "best.pt") for task in merging_tasks]
    ft_checks = [torch.load(fp) for fp in filepaths]
    ptm_check = pretrained_state_dict

    # check if all checkpoints have the same paramters.
    check_parameterNamesMatch(ft_checks + [ptm_check])
    remove_keys = [
        "transformer.encoder.embed_tokens.weight",
        "transformer.decoder.embed_tokens.weight",
    ]
    reshape_keys = ["transformer.shared.weight", "transformer.lm_head.weight"]
    for key in remove_keys + reshape_keys:
        ptm_check[key] = ptm_check[key][: len(ft_checks[0][key]), :]

    logger.info(f"Flattening out Checkpoints")
    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm

    # check if the vectorized state dicts can be converted back to the original state dicts.
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ft_checks[0], remove_keys), ptm_check
    )
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    )
    assert all(
        [
            check_state_dicts_equal(
                vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )

    logger.info(
        f"Pretrained Model: {base_model}\tMean:{flat_ptm.mean():.6f}\tSTD: {flat_ptm.std():.6f}"
    )
    logger.info(f"Merging tasks: {merging_tasks}")
    logger.info(
        f"Finetune checkpoint \nMean:{flat_ft.mean(dim=1)}\nSTD: {flat_ft.std(dim=1)}"
    )
    logger.info(
        f"Task Vector Finetune checkpoint\nMean:{tv_flat_checks.mean(dim=1)}\nSTD: {tv_flat_checks.std(dim=1)}"
    )

    return (
        tv_flat_checks,
        flat_ptm,
        flat_ft,
        ft_checks,
        ptm_check,
        merging_tasks,
        dump_dir,
        remove_keys,
        reshape_keys,
        base_model,
        load_dir,
        filepaths,
    )


def load_vit(merging_tasks, model_key):

    dump_dir = BASIC_INFORMATION[model_key]["dump_dir"]
    base_model = BASIC_INFORMATION[model_key]["base_model"]
    load_dir = BASIC_INFORMATION[model_key]["load_dir"]

    # load the finetune and pretrained checkpoints
    filepaths = [os.path.join(load_dir, task, "finetuned.pt") for task in merging_tasks]
    ft_checks = [torch.load(fp).state_dict() for fp in filepaths]
    ptm_check = torch.load(os.path.join(load_dir, "zeroshot.pt")).state_dict()

    # check if all checkpoints have the same paramters.
    check_parameterNamesMatch(ft_checks + [ptm_check])

    remove_keys = []
    reshape_keys = []
    for key in remove_keys + reshape_keys:
        ptm_check[key] = ptm_check[key][: len(ft_checks[0][key]), :]

    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm

    # check if the vectorized state dicts can be converted back to the original state dicts.
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ft_checks[0], remove_keys), ptm_check
    )
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    )
    assert all(
        [
            check_state_dicts_equal(
                vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )

    logger.info(
        f"Pretrained Model: {base_model}\tMean:{flat_ptm.mean():.6f}\tSTD: {flat_ptm.std():.6f}"
    )
    logger.info(f"Merging tasks: {merging_tasks}")
    logger.info(
        f"Finetune checkpoint \tMean:{flat_ft.mean(dim=1)}\tSTD: {flat_ft.std(dim=1)}"
    )
    logger.info(
        f"Task Vector Finetune checkpoint\tMean:{tv_flat_checks.mean(dim=1)}\tSTD: {tv_flat_checks.std(dim=1)}"
    )

    return (
        tv_flat_checks,
        flat_ptm,
        flat_ft,
        ft_checks,
        ptm_check,
        merging_tasks,
        dump_dir,
        remove_keys,
        reshape_keys,
        base_model,
        load_dir,
        filepaths,
    )
