import json
import os
import logging
import datetime

from shutil import copytree, ignore_patterns, rmtree

from src.utils.utils import safe_makedirs, parse_modelName, saveTo_gcp
from src.utils.Config import Config

logger = logging.getLogger("root")


class TrainingConfig(Config):
    def __init__(self, config_filepaths=None, kwargs=None, create_expDir=True):
        """
        Create config

        Args:
            config_filepaths:
            kwargs:
            create_expDir:
        """
        super().__init__()
        """
        dataset parameters
        """
        self.train_dataset = None
        self.inference_dataset = None

        self.train_dataset_mixture = None
        self.inference_dataset_mixture = None
        self.split = None

        self.max_datapoints_per_dataset = 500000
        self.max_datapoints_per_dataset_without_templates = None
        self.projection_loss = 0
        self.train_template_idx = -1
        self.eval_template_idx = -1
        self.few_shot_random_seed = None
        self.num_val_samples = 32
        self.evaluate_type = "inference" # inference, trainloss

        """
        model parameters
        """
        self.pretrained_model = "bigscience/T0_3B"
        self.max_seq_len = 512
        self.peft_method = None
        self.checkpoint_to_directly_load_model = None
        self.function_to_merge_checkpoints = None
        self.model_checkpoints = None
        self.trainable_param_regex = None

        # merging parameters
        self.model_lambda = None
        self.search_lambda = "none"
        self.temperature = 1
        self.learn_pretraining_lambda = False
        self.pretraining_init = 1
        self.lambda_init = 0.2
        self.lambda_type = (
            "vector-shared"  # scalar-shared, vector-shared, vector-notshared
        )
        self.pretrained_lambda_type = "scalar-shared"  # scalar-shared, scalar-notshared

        """
        training run parameters
        """
        # batch_size is per GPU and per gradient step.
        # The effective batch_size is train_batch_size * gradient_accumulation_factor * number_of_gpus
        self.train_batch_size = 32
        # num_batches is number of effective batches where each batch is
        # train_batch_size * gradient_accumulation_factor * number_of_gpus
        self.num_batches = 10000
        self.gradient_accumulation_factor = 1
        self.use_bfloat16_during_training = True

        """
        checkpoint parameters
        """
        self.resume_checkpoint_filepath = None
        self.checkpoint_frequency = 1000  # frequency is in terms of number of batches
        self.early_stopping = True
        self.early_stopping_num_checkpoints_without_improvement = 5
        self.should_save_most_recent_state = False
        self.should_save_every_checkpoint = False
        self.should_save_to_gcp = False
        self.should_eval_train = False
        self.should_eval_validation = True
        self.should_eval_at_beginning = False
        self.experiment_dir = None
        self.project_name = "debug"
        self.experiment_name = "test"
        self.delete_exist_dir = False

        """
        distributed training
        """
        self.world_size = None

        """
        evaluation parameters
        """
        self.max_gen_len = 64
        self.eval_batch_size = 64
        self.length_normalization = False
        self.use_bfloat16_during_eval = False

        """
        optimization parameters
        """
        self.lr = 1e-5
        self.optimizer = "adamw"
        self.scheduler = None
        self.warmup_ratio = 0.05
        self.weight_decay = 0.0
        self.norm_to_clip_gradient = None

        """
        reproducabilty parameters
        """
        self.seed = 42

        # Update config with values from list of files
        if config_filepaths:
            for filename in config_filepaths:
                super()._update_fromDict(
                    json.load(open(filename)), assert_keyInUpdateDict_isValid=True
                )

        # Update config with values from kwargs
        if kwargs:
            super()._update_fromDict(kwargs, assert_keyInUpdateDict_isValid=False)

        # Create new experiment directory
        if create_expDir:
            if self.resume_checkpoint_filepath is not None:
                logger.info(
                    "Warning: Since we are resuming from checkpoint, no new experiment "
                    "directory is created "
                )
                return

            base_dir = "exp_out"
            if self.project_name is not None:
                base_dir = os.path.join(base_dir, self.project_name)

            # Parse the model name to create directory from model name
            pretrained_model = parse_modelName(self.pretrained_model)
            pretrained_model = (
                f"{self.peft_method}-{pretrained_model}"
                if self.peft_method is not None
                else pretrained_model
            )
            base_dir = os.path.join(base_dir, pretrained_model)

            if self.experiment_name is not None:
                base_dir = os.path.join(base_dir, self.experiment_name)

            # base_dir = self.create_expDir(base_dir)
            if os.path.exists(base_dir) and self.delete_exist_dir:
                print(f"Deleting Existing directory {base_dir}".upper())
                rmtree(base_dir)

            os.makedirs(base_dir, exist_ok=True)

            self.experiment_dir = base_dir
            self.save_config(os.path.join(self.experiment_dir, "config.json"))

    def create_expDir(self, base_dir):
        if os.path.exists(base_dir) and not self.exp_dir_exist_ok:
            raise ValueError(f"Experiment directory {base_dir} already exists")
        else:
            i = 0
            while True:
                temp_base_dir = f"{base_dir}~try={i}"
                if not os.path.exists(temp_base_dir):
                    os.makedirs(temp_base_dir)
                    base_dir = temp_base_dir
                    break
                i += 1
        return base_dir

    def save_config(self, config_fp):
        """
        Save config at filename

        Args:
            filename:

        Returns:

        """
        with open(config_fp, "w+") as f:
            f.write(json.dumps(self.__dict__, indent=4, sort_keys=True))
            f.write("\n")

        saveTo_gcp(self.should_save_to_gcp, config_fp)
