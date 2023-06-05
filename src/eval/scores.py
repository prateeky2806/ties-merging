import json

from src.utils.utils import get_average, get_median, get_interquartileRange

from src.data.dataset_mixtures import get_datasetMixture

from src.utils.NoIndentEncoder import NoIndentEncoder, noIndent_dictOrList_onFirstLevel

from src.utils.utils import check_string_equality


def get_whichValue_dictDiffers(dict_A, dict_B):
    """
    Assumes only 1 value differs between dictionary and finds that value.

    Args:
        dict_A:
        dict_B:

    Returns:

    """
    assert dict_A.keys() == dict_B.keys()

    keys = dict_A.keys()

    key_thatDiffers = None
    value_A = None
    value_B = None

    for key in keys:

        if dict_A[key] != dict_B[key]:

            if key_thatDiffers is not None:
                raise ValueError("More than 1 Key Differs between dictionaries")

            key_thatDiffers = key
            value_A = dict_A[key]
            value_B = dict_B[key]

    return key_thatDiffers, value_A, value_B


def concatenate_scores(multiple_configAndScores, default_key="inference_dataset"):
    """
    Concatenate all the scores for different configs into a dictionary, where a key is the value that
    differs.

    Args:
        multiple_configAndScores:
        default_key: key of config to use

    Returns:

    """
    if len(multiple_configAndScores) == 1:
        first_configAndScores = multiple_configAndScores[0]
        assert (
            default_key is not None
        ), "default_key cannot be None if the number of scores is 0."
        return {
            first_configAndScores["config"][default_key]: first_configAndScores["score"]
        }

    previous_configAndScore = multiple_configAndScores[0]
    next_configAndScore = multiple_configAndScores[1]

    key_thatDiffers, value_A, value_B = get_whichValue_dictDiffers(
        previous_configAndScore["config"], next_configAndScore["config"]
    )

    if default_key is not None and default_key != "inference_dataset":
        assert key_thatDiffers == default_key
    concatenated_scores = {
        value_A: previous_configAndScore["score"],
        value_B: next_configAndScore["score"],
    }

    for config_and_score in multiple_configAndScores[1:]:
        currentKey_thatDiffers, _, value_B = get_whichValue_dictDiffers(
            previous_configAndScore["config"], config_and_score["config"]
        )

        check_string_equality(key_thatDiffers, currentKey_thatDiffers)
        concatenated_scores[value_B] = config_and_score["score"]

    return concatenated_scores


def get_averageScores(multiple_configAndScores):
    """

    Args:
        multiple_configAndScores:

    Returns:

    """
    individual_averageScores = list(
        map(lambda x: x["score"]["average"], multiple_configAndScores)
    )
    average_score = get_average(individual_averageScores)

    return {"average": average_score}


def extact_score(multiple_configAndScores):
    """

    Args:
        multiple_configAndScores:

    Returns:

    """
    assert len(multiple_configAndScores) == 1

    return multiple_configAndScores[0]["score"]


def get_summaryOfScores_acrossPrompts(multiple_configAndScores):
    """

    Args:
        multiple_configAndScores:

    Returns:

    """
    individual_averageScores = list(
        map(lambda x: x["score"]["average"], multiple_configAndScores)
    )

    summary_ofScores = {
        "median": get_median(individual_averageScores),
        "interquartile_range": get_interquartileRange(individual_averageScores),
        "average_scores_for_each_prompt": individual_averageScores,
    }

    return summary_ofScores


def saveResult_acrossLabels(labels, scores, getScore_fn, score_fp):
    """
    Save the average of the average score for each dataset

    Args:
        labels:
        scores:
        getScore_fn:
        score_fp:

    Returns:

    """
    scores_toDisplay = []

    for label in labels:
        scores_toDisplay.append(getScore_fn(scores[label]))

    label_str = ",".join(labels)
    scores_str = ",".join(scores_toDisplay)

    print(label_str)
    print(scores_str)

    with open(score_fp, "a+") as f:
        f.write(label_str + "\n")
        f.write(scores_str + "\n")


def saveResult_acrossDatasetMixture(
    dataset_mixture, scores, getScore_fn, score_fp, saveAverage_acrossDatasets
):
    """
    Save the average of the average score for each dataset

    Args:
        dataset_mixture:
        scores:
        getScore_fn:
        score_fp:
        saveAverage_acrossDatasets:

    Returns:

    """
    labels_toDisplay = []
    scores_toDisplay = []

    if saveAverage_acrossDatasets:
        labels_toDisplay.append("Avg.")
        scores_toDisplay.append(str(round(scores["average"] * 100, 1)))

    if dataset_mixture is not None:
        labels_toDisplay.extend(get_datasetMixture(dataset_mixture))

        for dataset in get_datasetMixture(dataset_mixture):
            scores_toDisplay.append(getScore_fn(scores[dataset]))

    label_str = ",".join(labels_toDisplay)
    scores_str = ",".join(scores_toDisplay)

    print(label_str)
    print(scores_str)

    with open(score_fp, "a+") as f:
        f.write(label_str + "\n")
        f.write(scores_str + "\n")


def save_allScores(scores, score_fp):
    """

    Args:
        scores:
        score_fp:

    Returns:

    """
    with open(score_fp, "a+") as f_out:
        f_out.write(
            json.dumps(
                noIndent_dictOrList_onFirstLevel(scores), cls=NoIndentEncoder, indent=2
            )
            + "\n"
        )
