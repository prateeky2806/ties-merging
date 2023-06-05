import sys, os
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import logging
import torch
import math
import numpy as np
import re
from collections import OrderedDict
from scipy.stats import wasserstein_distance
import torch.nn.functional as F

logger = logging.getLogger("root")


def browse_all_files_in_a_directory(directory, file_extension):
    """Returns a list of all files in a directory with a given extension"""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(file_extension)]

def walk_through_all_files_in_a_directory(directory, file_extension):
    """Returns a list of all files in a directory with a given extension"""
    return [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if file.endswith(file_extension)]

def extract_number(filename):
    # Regular expression to extract a number from a string
    return int(filename.split("_")[-1].split(".")[0])

def all_pairwise_cosine_similarity(vectors):
    """
    Returns True if the given set of vectors are orthogonal, False otherwise.

    Args:
    - vectors (tensor): A tensor of shape (n, d) representing n d-dimensional vectors.
    - tolerance (float): The tolerance level for checking if the dot product is close to zero.

    Returns:
    - True if the vectors are orthogonal, False otherwise.
    """
    # Compute the dot product of every pair of vectors
    num_vectors = vectors.shape[0]
    dot_products = torch.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            cosine_sim = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))
            dot_products[i, j] = cosine_sim

    # If all dot products are close to zero, return True
    return dot_products

def get_avg_pairwise_cosine_similarity(matrix):
    if not torch.is_tensor(matrix):
        matrix = torch.tensor(matrix)
    cos_sims = all_pairwise_cosine_similarity(matrix)
    mm = torch.tril(cos_sims, diagonal=-1)
    dim = len(cos_sims)
    non_zeros = ((dim-1)*(dim))/2
    return mm.sum() / non_zeros

def state_dict_to_vector(state_dict):
    """
    Convert a PyTorch state dict to a flattened parameter vector.

    Args:
        state_dict (dict): The state dict to convert.

    Returns:
        torch.Tensor: The flattened parameter vector.
    """
    sorted_dict = OrderedDict(sorted(state_dict.items()))
    return torch.nn.utils.parameters_to_vector(sorted_dict.values())

def vector_to_state_dict(vector, state_dict):
    """
    Convert a flattened parameter vector to a PyTorch state dict.

    Args:
        vector (torch.Tensor): The flattened parameter vector.
        state_dict (dict): The original state dict with parameter shapes.

    Returns:
        dict: The new state dict with parameters in the correct shape.
    """
    new_state_dict = state_dict.copy()
    sorted_dict = OrderedDict(sorted(new_state_dict.items()))
    torch.nn.utils.vector_to_parameters(vector, sorted_dict.values())
    return sorted_dict

def change_key_to_name(dict):
    new_dict = OrderedDict()
    for k,v in dict.items():
        name = k.split('/')[-2]
        new_dict[name] = v
    return new_dict