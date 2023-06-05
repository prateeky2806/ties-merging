import os
import logging
import copy
import random
import datasets
import json

from promptsource.templates import DatasetTemplates, Template
from datasets import load_dataset

logger = logging.getLogger("root")
# datasets.disable_progress_bar()


class DatasetReader(object):
    """
    DatasetReader objects reads dataset and has all attributes specific to dataset
    """

    def __init__(self, dataset_stash, template_stash):

        self.dataset_stash = dataset_stash
        self.template_stash = template_stash

        self.all_templates = self._get_datasetTemplates(None, None)

        self.cached_origData = {}
        self.cached_datasets = {}

    def _get_origData(self, split):
        """
        Reads the original dataset split from huggingface. Converts the label to an int and returns the updated dataset.
        Args:
            split:

        Returns:

        """

        if self.few_shot_random_seed is not None:
            return self._read_few_shot_dataset(split, self.few_shot_random_seed)
        else:
            return self._read_origin_dataset(split)

    def _read_origin_dataset(self, split):
        """
        Reads the original dataset split from huggingface. Converts the label to an int and returns the updated dataset.
        Args:
            split:

        Returns:

        """
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            logger.info(f"\t\tLoading Full Data for {self.name}")
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )
            orig_data = []
            # converting label to int and caching the split of the dataset.
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                if len(orig_data) > self.num_val_samples:
                    orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )
                else:
                    print(f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}")
                    print("splitting equally")
                    orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                        orig_data, len(orig_data)//2
                    )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _read_few_shot_dataset(
        self,
        split,
        few_shot_random_seed,
    ):
        if split not in self.cached_origData:
            logger.info(
                f"\t\tLoading Few Shot Data for {self.name} with seed {few_shot_random_seed}"
            )
            file_path = os.path.join(
                "data",
                "few_shot",
                self.name,
                f"{few_shot_random_seed}_seed.jsonl",
            )
            if os.path.exists(file_path):
                with open(file_path, "r") as fin:
                    data = []
                    for idx, line in enumerate(fin.readlines()):
                        example = json.loads(line.strip("\n"))
                        example["lbl"] = int(example["label"])
                        data.append(example)
                    self.cached_origData[split] = data
            else:
                raise ValueError(f"Few shot dataset not found at {file_path}")

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        """
        Returns a list of all templates for the dataset with the given metrics and not in the list of templates to ignore.
        Args:
            templateNames_toIgnore:
            metrics_toUse: specify the metric to use so that we only include templates which
                           match the metric we want to use

        Returns:

        """
        all_templates = []

        # Get original templates from promptsource
        for template in DatasetTemplates(*self.template_stash).templates.values():
            # Filter out templates that
            # 1) are not designed for original task
            # 2) have different metrics than we want to use
            # 3) are ones that we want to ignore based on the name
            if template.metadata.original_task:
                should_ignoreTemplate = False

                for metric in template.metadata.metrics:
                    if metric not in metrics_toUse:
                        should_ignoreTemplate = True

                for template_name in templateNames_toIgnore:
                    if template.name == template_name:
                        should_ignoreTemplate = True

                if not should_ignoreTemplate:
                    all_templates.append(template)

        return all_templates

    def _applyTemplate_toData(
        self, orig_data, num_templates, template_idx, is_evaluation
    ):
        """
        Args:
            orig_data:
            num_templates:
            template_idx:
            is_evaluation:

        Returns:

        """
        dataset = []

        for datapoint_idx, datapoint in enumerate(orig_data):

            # Use fixed template across entire dataset
            if template_idx >= 0:
                templateIdx_forDatapoint = template_idx

            # Use all templates across entire dataset, where different datapoints can get
            # different templates. However, a datapoint is always matched with the same template
            elif template_idx == -1:
                templateIdx_forDatapoint = datapoint_idx % num_templates

            # select a random template for the example.
            elif template_idx == -3:
                templateIdx_forDatapoint = random.randint(0, len(self.all_templates))

            else:
                raise ValueError(f"Invalid template index {templateIdx_forDatapoint}")

            template = self.all_templates[templateIdx_forDatapoint]
            new_datapoint = copy.deepcopy(datapoint)

            # For evaluation, we add the answer_choices if they exist
            if is_evaluation:
                answer_choices = template.get_answer_choices_list(datapoint)
                if answer_choices is not None:
                    new_datapoint["answer_choices"] = answer_choices

            # We apply the template to datapoint instead of new_datapoint since the answer_choices
            # are added in the template function, and so applying the template to new_datapoint
            # will cause an error with the answer_choices key
            input_txt, target_txt = template.apply(datapoint)
            new_datapoint["input"] = input_txt

            # For non-evaluation or tasks where they are no answer_choices, we just add the target (
            # the correct answer_choice)
            if not is_evaluation or "answer_choices" not in new_datapoint:
                new_datapoint["target"] = target_txt

            dataset.append(new_datapoint)

        return dataset

    def _split_val_into_val_and_test(self, orig_data, num_val_samples=32, seed=42):
        """
        Splits the validation set into validation and test set. This is done by taking the first 1000 examples
        as the test set and the rest as the validation set.
        Args:
            orig_data:
            seed:

        Returns:

        """
        random.seed(seed)
        random.shuffle(orig_data)
        val_data = orig_data[:num_val_samples]
        test_data = orig_data[num_val_samples:]
        return val_data, test_data

    def get_dataset(
        self, split, template_idx, is_evaluation, max_samples_per_dataset=None
    ):
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
                if ==-3, apply a random template to each datapoint.
            is_evaluation: whether the split is for evaluation (where it will have answer_choices)
                            or for training (where it will only have the target)
        Returns:
            dataset:
        """
        if (split, template_idx) not in self.cached_datasets:
            orig_data = self._get_origData(split)
            total_examples = len(orig_data)
            orig_data = (
                orig_data[: self.max_datapoints_per_dataset_without_templates]
                if self.max_datapoints_per_dataset_without_templates
                and split.lower() == "train"
                else orig_data
            )
            logger.info(
                f"\tDataset:{self.name.upper()}\tSplit:{split}\tSelected Examples: {len(orig_data)}\tNum Total Example:{total_examples}"
            )
            num_templates = self.get_numTemplates()

            # template_idx -2 means we do a cross product of each datapoint with each template
            if template_idx == -2:
                dataset = []
                for iterate_templateIdx in range(num_templates):
                    dataset.extend(
                        self._applyTemplate_toData(
                            orig_data, num_templates, iterate_templateIdx, is_evaluation
                        )
                    )
            # otherwise apply template to dataset
            else:
                dataset = self._applyTemplate_toData(
                    orig_data, num_templates, template_idx, is_evaluation
                )
            # shuffle examples and select max_samples perdataset as same examples with different templates will occur together.
            random.Random(4).shuffle(dataset)
            total_examples_with_templates = len(dataset)
            dataset = (
                dataset[:max_samples_per_dataset]
                if max_samples_per_dataset
                else dataset
            )
            logger.info(
                f"\tDataset:{self.name.upper()}\tSplit:{split}\tNum Selected Example with Templates:{len(dataset)}\tTemplate Idx:{template_idx}\tNum Templates:{num_templates}\tNum Examples with Template:{total_examples_with_templates}"
            )

            self.cached_datasets[(split, template_idx)] = dataset

        return self.cached_datasets[(split, template_idx)]

    def get_numTemplates(self):
        return len(self.all_templates)

    def get_metricsForDataset(self):
        return self.all_templates[0].metadata.metrics


class RTEReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("super_glue", "rte"), template_stash=("super_glue", "rte")
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "rte"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class HSwagReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("hellaswag",), template_stash=("hellaswag",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "h-swag"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        all_templates = super()._get_datasetTemplates(
            ["Randomized prompts template"], ["Accuracy"]
        )

        # Add each template from the several templates in the randomized prompt individually
        listOf_randomJinjas = [
            (
                "randomized prompt 1",
                "Can you pick the correct ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 2",
                "The task is to generate the ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 3",
                "How does this sentence end? {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 4",
                "From the list of endings described below, what ending makes the most sense for the sentence {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
        ]

        for name, jinja in listOf_randomJinjas:
            all_templates.append(
                Template(
                    name=name,
                    jinja=jinja,
                    reference="",
                    answer_choices='{{endings | join("|||")}}',
                )
            )

        return all_templates


class COPAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("super_glue", "copa"), template_stash=("super_glue", "copa")
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "copa"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates(
            [
                "…which may be caused by",
                "…What could happen next, C1 or C2?",
                "…As a result, C1 or C2?",
                "…why? C1 or C2",
            ],
            ["Accuracy"],
        )


class WiCReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("super_glue", "wic"), template_stash=("super_glue", "wic")
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "wic"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class WinograndeReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("winogrande", "winogrande_xl"),
            template_stash=("winogrande", "winogrande_xl"),
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "winogrande"

    def _read_origin_dataset(self, split):
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["answer"]) - 1
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class CBReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("super_glue", "cb"), template_stash=("super_glue", "cb")
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "cb"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class StoryClozeReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("story_cloze", "2016"),
            template_stash=("story_cloze", "2016"),
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "story_cloze"

    def _read_origin_dataset(self, split):

        # We use the test set of StoryCloze for validation and the validation set of StoryCloze
        # for train - following GPT3
        if split == "train":
            load_split = "validation"
        elif split in ["validation", "validation_full", "test"]:
            load_split = "test"

        if split not in self.cached_origData:
            # Do not use default method for loading dataset since the story_cloze dataset must be
            # downloaded manually and then we have to set data_dir to point to it.
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
                data_dir="",
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["answer_right_ending"]) - 1
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class ANLIR1Reader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("anli",), template_stash=("anli",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "anli-r1"

    def _read_origin_dataset(self, split):

        load_split = "dev" if "validation" in split.lower() else split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=f"{load_split}_r1",
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "validation":
                random.seed(42)
                random.shuffle(orig_data)
                orig_data = orig_data[: self.num_val_samples]
            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class ANLIR2Reader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("anli",), template_stash=("anli",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "anli-r2"

    def _read_origin_dataset(self, split):

        load_split = "dev" if "validation" in split.lower() else split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=f"{load_split}_r2",
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "validation":
                random.seed(42)
                random.shuffle(orig_data)
                orig_data = orig_data[: self.num_val_samples]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class ANLIR3Reader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("anli",), template_stash=("anli",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "anli-r3"

    def _read_origin_dataset(self, split):

        load_split = "dev" if "validation" in split.lower() else split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=f"{load_split}_r3",
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "validation":
                random.seed(42)
                random.shuffle(orig_data)
                orig_data = orig_data[: self.num_val_samples]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class WSCReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("super_glue", "wsc.fixed"),
            template_stash=("super_glue", "wsc.fixed"),
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "wsc"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class CosmosQAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("cosmos_qa",), template_stash=("cosmos_qa",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "cosmos_qa"

    def _read_origin_dataset(self, split):

        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class SocialIQAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("social_i_qa",), template_stash=("social_i_qa",)
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "social_i_qa"

    def _read_origin_dataset(self, split):

        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"]) - 1
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates(
            ["Check if a random answer is valid or not"], ["Accuracy"]
        )


class PAWSReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("paws", "labeled_final"),
            template_stash=("paws", "labeled_final"),
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "paws"

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = example["label"]
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class QuAILReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("quail",), template_stash=("quail",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "quail"

    def _read_origin_dataset(self, split):

        load_split = "challenge" if split == "test" else split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = example["correct_answer_id"]
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class WikiQAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("wiki_qa",), template_stash=("wiki_qa",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "wiki_qa"

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class QuaRTzReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("quartz",), template_stash=("quartz",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "quartz"

        self.string_toLabelIdx = {"A": 0, "B": 1}

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class QASCReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("qasc",), template_stash=("qasc",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "qasc"

        self.string_toLabelIdx = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
        }

    def _read_origin_dataset(self, split):

        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class ROPESReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(dataset_stash=("ropes",), template_stash=("ropes",))

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)

        self.name = "ropes"

    def _read_origin_dataset(self, split):

        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["answers"]["answer_start"] = [0]
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Squad"])


DATASET_CLASSES = {
    "rte": RTEReader,
    "h-swag": HSwagReader,
    "copa": COPAReader,
    "wic": WiCReader,
    "winogrande": WinograndeReader,
    "cb": CBReader,
    "story_cloze": StoryClozeReader,
    "anli-r1": ANLIR1Reader,
    "anli-r2": ANLIR2Reader,
    "anli-r3": ANLIR3Reader,
    "wsc": WSCReader,
    "cosmos_qa": CosmosQAReader,
    "social_iqa": SocialIQAReader,
    "paws": PAWSReader,
    "quail": QuAILReader,
    "wiki_qa": WikiQAReader,
    "quartz": QuaRTzReader,
    "qasc": QASCReader,
    "ropes": ROPESReader,
}


def get_datasetReader(dataset_name, dataset_kwargs=None):
    return DATASET_CLASSES[dataset_name](dataset_kwargs)
