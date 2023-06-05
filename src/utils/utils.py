import random, sys
import numpy as np
import os
import argparse
import subprocess
import statistics
import torch
import copy
import json
import logging
import logging.config
import wandb
from pathlib import Path

from statistics import mean
from scipy.stats import iqr

logger = logging.getLogger("root")


def setup_wandb_logger(config):
    wandb_writer = wandb.init(
        project=config["project_name"],
        entity="user",
        save_code=False,
        name=config["experiment_name"],
        config=config,
        dir=config["experiment_dir"],
    )  # , group=config.group)

    src_dir = Path(__file__).resolve().parent
    base_path = str(src_dir.parent)
    src_dir = str(src_dir)
    return wandb_writer, src_dir, base_path


def set_seeds(seed):
    """
    Set all random seeds to the fixed seed

    Args:
        seed:

    Returns:

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_listOfDict_toDictOfList(list_ofDict):
    """
    Args:
        list_ofDict:

    Returns:
        dict_ofList
    """
    dict_ofList = {}

    for single_dict in list_ofDict:
        for (k, v) in single_dict.items():
            if k in dict_ofList:
                dict_ofList[k].append(v)
            else:
                dict_ofList[k] = [v]

    return dict_ofList


# From https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
def convert_dictOfLists_to_listOfDicts(dictOfLists):
    listOfDicts = []
    for datapoint_values in zip(*dictOfLists.values()):
        listOfDicts.append(dict(zip(dictOfLists, datapoint_values)))
    return listOfDicts


def safe_makedirs(dir_name):
    """
    Makes a directory if it doesn't exists yet

    Args:
        dir_name: directory name
    """
    exist = None
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        exist = False
    else:
        logger.info(f"Directory {dir_name} already exists")
        exist = True
    return exist


def getValueOfKey_inDictionary(dictionary_toSearch, keys_toSearchFor):
    """
    Check if key or path of key exists in dictionary and return the value correspoding to the key

    Args:
        dictionary_toSearch:
        keys_toSearchFor: returns the value of the first key that is found in dictionary

    Returns:

    """

    for full_key in keys_toSearchFor:
        # Full key can be path in nested dictionary
        if isinstance(full_key, tuple):
            for key in full_key:
                # If key exists in dictionary, keep searching deeper
                if key in dictionary_toSearch:
                    dictionary_toSearch = dictionary_toSearch[key]

                    # If found value, return it
                    if not isinstance(dictionary_toSearch, dict):
                        return dictionary_toSearch
                    # Continue searching children dictionary_toSearch
                    else:
                        continue
                # Else skip to next key
                else:
                    continue

        else:
            # If key exists in dictionary, return it
            if full_key in dictionary_toSearch:
                dictionary_toSearch = dictionary_toSearch[full_key]

                # If found value, return it
                if not isinstance(dictionary_toSearch, dict):
                    return dictionary_toSearch
                else:
                    raise ValueError(
                        "Key specifies dictionary not value", dictionary_toSearch
                    )
            # Else skip to next key
            else:
                continue

    raise ValueError("None of the keys found", dictionary_toSearch)


class ParseKwargs(argparse.Action):
    """
    Parse Kwargs into dictionary
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        if len(values) == 1:
            values = values[0].split()
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def parse_modelName(model_name):
    """
    Removes any directory prefix in model_name and replace / with -

    Args:
        model_name:

    Returns:

    """
    return model_name.replace("/fruitbasket/models/", "").replace("/", "-")


def saveTo_gcp(should_saveToGCP, filepath):
    """

    Args:
        should_saveToGCP:
        filepath:

    Returns:

    """
    if should_saveToGCP:
        subprocess.call(
            f"gsutil "
            f"-m "
            f"-o GSUtil:parallel_composite_upload_threshold=150M "
            f"cp -r {filepath} gs://merging_models_via_loss_landscape/{filepath}",
            shell=True,
        )


def get_median(list_ofNumbers):
    """


    Args:
        all_scores: list of dictionaries, where one of the value is the score we are interested in

    Returns:

    """
    return round(statistics.median(list_ofNumbers), 3)


def get_interquartileRange(list_ofNumbers):
    """


    Args:
        list_ofNumbers:

    Returns:

    """
    return round(iqr(list_ofNumbers), 3)


def get_average(list_ofNumbers):
    """

    Args:
        list_ofNumbers:

    Returns:

    """
    return round(mean(list_ofNumbers), 3)


def round_list(my_list, significant_figures):
    """

    Args:
        list:
        significant_figures:

    Returns:

    """
    rounded_list = []

    for number in my_list:
        rounded_list.append(round(number, significant_figures))

    return rounded_list


def round_nestedList(nested_list, significant_figures):
    """
    Round nested list of numbers where list can be any depth

    Args:
        nested_list:
        significant_figures:

    Returns:
        round_nestedList
    """
    rounded_nestedList = []
    for sublist in nested_list:

        if isinstance(sublist[0], list):
            rounded_sublist = round_nestedList(sublist, significant_figures)
        else:
            rounded_sublist = round_list(sublist, significant_figures)

        rounded_nestedList.append(rounded_sublist)

    return rounded_nestedList


def read_jsonl(filepath):
    """
    Read JSONL filepath

    Args:
        filepath:

    Returns:
    """
    json_lines = []

    with open(filepath, "r") as f:
        for idx, line in enumerate(f.readlines()):
            json_lines.append(json.loads(line.strip("\n")))

    return json_lines


# From https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200
def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def group_by(list_ofItems, fn_toGetGroupByField):
    groups = {}

    for my_dict in list_ofItems:
        field = fn_toGetGroupByField(my_dict)

        if field in groups:
            groups[field].append(my_dict)
        else:
            groups[field] = [my_dict]

    return groups


def map_forDictionaries(my_dict, map_fn):
    mapped_dict = {}
    for (k, v) in my_dict.items():
        mapped_dict[k] = map_fn(v)
    return mapped_dict


def breadth_first_search(values_toIterateOver):
    """

    Args:
        values_toIterateOver:

    Returns:

    """

    if isinstance(values_toIterateOver, list):
        return list(map(lambda x: [x], values_toIterateOver))

    all_possibilities = []

    for key in values_toIterateOver.keys():
        sub_possibilities = breadth_first_search(values_toIterateOver[key])

        for possibility in sub_possibilities:
            copy_possibility = copy.deepcopy(possibility)
            copy_possibility.insert(0, key)
            all_possibilities.append(copy_possibility)

    return all_possibilities


def get_logger(name, log_dir, config_path):
    """
    Creates a logger object
    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read
    Returns
    -------
    A logger object which writes to both file and stdout
    """
    config_dict = json.load(open(config_path))
    config_dict["handlers"]["file_handler"]["filename"] = log_dir + name.replace(
        "/", "-"
    )
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger("root")

    std_out_format = "%(asctime)s - [%(levelname)s] - %(message)s"
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


import torch


def check_string_equality(string_one, string_two):
    assert string_one == string_two, (
        f"\n{string_one}\n" + "=" * 100 + "\n" + f"{string_two}"
    )


def check_string_subset_of_another(string_one, string_two):
    assert string_one in string_two, (
        f"\n{string_one}\n" + "=" * 100 + "\n" + f"{string_two}"
    )


def check_string_starts_with_another(string_one, string_two):
    assert string_one.startswith(string_two), (
        f"\n{string_one}\n" + "=" * 100 + "\n" + f"{string_two}"
    )


def check_string_ends_with_another(string_one, string_two):
    assert string_one.endswith(string_two), (
        f"\n{string_one}\n" + "=" * 100 + "\n" + f"{string_two}"
    )


def check_list_equality(list_one, list_two):

    if list_one != list_two:
        print(f"\n{list_one}\n" + "=" * 100 + "\n" + f"{list_two}")

        if len(list_one) != len(list_two):
            print(
                f"The lengths differ. \n"
                f"list_one has length {len(list_one)} \n"
                f"list_two has lengths {len(list_two)}"
            )

        for (idx, (element_one, element_two)) in enumerate(zip(list_one, list_two)):
            if element_one != element_two:
                print(
                    f"The first element that differs is at position {idx}. \n"
                    f"list_one has element {element_one} \n"
                    f"list_two has element {element_two}"
                )
                return False

    return True


def checkEqual_modelParameters(modelOne_parameters, modelTwo_parameters):
    for parameter_name, parameter in modelOne_parameters.items():
        assert torch.allclose(parameter, modelTwo_parameters[parameter_name], atol=1e-5)
