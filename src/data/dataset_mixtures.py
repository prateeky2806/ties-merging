import random
import logging

from src.data.dataset_readers import get_datasetReader, DATASET_CLASSES

logger = logging.getLogger("root")

EIGHT_QA_DATASETS = [
    "cosmos_qa",
    "social_iqa",
    "paws",
    "quail",
    "wiki_qa",
    "quartz",
    "qasc",
    "ropes",
]

T0_HELD_OUT_DATASETS = [
    "rte",
    "cb",
    "winogrande",
    "wic",
    "wsc",
    "copa",
    "h-swag",
    "story_cloze",
    "anli-r1",
    "anli-r2",
    "anli-r3",
]

T5_MIXTURE = ["anli-r1", "anli-r2", "anli-r3", "paws", "qasc", "quartz", "story_cloze", "wiki_qa", "winogrande", "wsc"]
T5_MIXTURE = ["paws", "qasc", "quartz", "story_cloze", "wiki_qa", "winogrande", "wsc"]


DATASET_MIXTURES = {
    "eight_qa": EIGHT_QA_DATASETS,
    "T0_held_out": T0_HELD_OUT_DATASETS,
    "t5_mixture": T5_MIXTURE,
    "t5_mixture": T5_MIXTURE,
}

class DatasetMixtureReader(object):
    def __init__(self, mixture_ofDatasetReaders, maximumDatapoints_perDataset):
        self.mixture_ofDatasetReaders = mixture_ofDatasetReaders
        self.maximumDatapoints_perDataset = maximumDatapoints_perDataset

        self.cached_origData = {}
        self.cached_datasets = {}

    def get_dataset(self, split, template_idx, is_evaluation):
        """
        Create dataset that includes the template

        Args:
            split:
            template_idx:
                if >=0, then we use the fixed template_idx across entire dataset
                if ==-1, then we use all template across entire the dataset, where different
                         datapoints can have different templates. A datapoint will always be
                         mapped to the same template though
                if ==-2, then we take the cross product of all templates and all datapoints.
                if ==-3, random template for each datapoint.

        Returns:
            dataset:
        """
        if (split, template_idx) not in self.cached_datasets:
            mixture_dataset = []

            for _, dataset_reader in self.mixture_ofDatasetReaders.items():

                # If we only use 1 template(), then get the dataset for each template
                if (template_idx >= -1) or template_idx == -3:
                    dataset = dataset_reader.get_dataset(
                        split,
                        template_idx,
                        is_evaluation,
                        self.maximumDatapoints_perDataset,
                    )
                    mixture_dataset.extend(dataset)

                else:
                    assert template_idx == -2

                    num_templates = dataset_reader.get_numTemplates()

                    maximumDatapoints_perDatasetAndTemplate = (
                        self.maximumDatapoints_perDataset // num_templates
                    )
                    # To ensure each dataset gets the same number of datapoints, if the maximum
                    # number of datapoints is not divisble by the number of templates, we compute
                    # the remainder and add back in this many datapoints
                    remainderDatasets_withExtraDatapoint = (
                        self.maximumDatapoints_perDataset % num_templates
                    )

                    # Add the same number of datapoints per template
                    for iterated_templateIdx in range(num_templates):
                        dataset = dataset_reader.get_dataset(
                            split, iterated_templateIdx, is_evaluation
                        )
                        # Fix the seed so that the examples chosen in the mixture of dataset is deterministic
                        random.seed(0)
                        random.shuffle(dataset)
                        mixture_dataset.extend(
                            dataset[:maximumDatapoints_perDatasetAndTemplate]
                        )

                        # if (iterated_templateIdx < remainderDatasets_withExtraDatapoint) and dataset_reader.max_datapoints_per_dataset_without_templates is None:
                        #     mixture_dataset.append(
                        #         dataset[maximumDatapoints_perDatasetAndTemplate]
                        #     )

            self.cached_datasets[(split, template_idx)] = mixture_dataset

        logger.info(
            f"\n\nNumber of datapoints in {split} for Mixture Dataset: {len(self.cached_datasets[(split, template_idx)])}\n\n"
        )

        return self.cached_datasets[(split, template_idx)]

    def get_numTemplates(self):
        raise ValueError("Cannot get number of templates for mixture of datasets")

    def get_metricsForDataset(self):
        raise ValueError("Cannot get metrics for mixture of datasets")


def get_datasetMixture(dataset_mixture):
    """

    Args:
        dataset_mixture:

    Returns:

    """
    # If dataset_mixture is a list, then check that each dataset in list is valid.
    if isinstance(dataset_mixture, list):
        for dataset in dataset_mixture:
            assert dataset in DATASET_CLASSES.keys()
        return dataset_mixture
    # If dataset_mixture is a string, then we look up the dataset mixture.
    else:
        assert dataset_mixture in DATASET_MIXTURES.keys()
        return DATASET_MIXTURES[dataset_mixture]


def get_datasetMixtureReader(
    datast_mixture, maxDatapoints_perDataset, dataset_kwargs=None
):
    """

    Args:
        datast_mixture:
        maxDatapoints_perDataset:

    Returns:
        mixtureDataset_reader:

    """
    mixture_ofDatasetReaders = {}

    for dataset in get_datasetMixture(datast_mixture):
        mixture_ofDatasetReaders[dataset] = get_datasetReader(dataset, dataset_kwargs)

    mixtureDataset_reader = DatasetMixtureReader(
        mixture_ofDatasetReaders, maxDatapoints_perDataset
    )

    return mixtureDataset_reader, mixture_ofDatasetReaders


def get_dataset_from_argparse(all_DatasetMixtures):
    datasets = []
    for dataset_or_mixture in all_DatasetMixtures:
        if dataset_or_mixture in DATASET_CLASSES.keys():
            datasets.append(dataset_or_mixture)
        elif dataset_or_mixture in DATASET_MIXTURES.keys():
            datasets.extend(DATASET_MIXTURES[dataset_or_mixture])
        else:
            raise ValueError(
                f"Invalid dataset or dataset mixture: {dataset_or_mixture}"
            )
    return datasets
