from src.utils.Config import Config


class EvaluationConfig(Config):
    def __init__(
        self, configDict_toInitializeFrom=None, fields_toUpdate=None, kwargs=None
    ):
        """

        Args:
            configDict_toInitializeFrom:
            fields_toUpdate:
            kwargs:
        """
        super().__init__()

        self.inference_dataset = None

        self.split = "validation"
        self.few_shot_random_seed = None
        self.num_val_samples = 32
        self.max_datapoints_per_dataset_without_templates = None

        self.should_save_to_gcp = False
        self.prediction_dir = None

        self.max_gen_len = 64
        self.eval_batch_size = 64
        self.eval_template_idx = -1

        self.length_normalization = False
        self.use_bfloat16_during_eval = False

        self.world_size = None

        self.did_run_finish = False

        if configDict_toInitializeFrom is not None:
            super()._update_fromDict(
                configDict_toInitializeFrom, assert_keyInUpdateDict_isValid=False
            )

        if fields_toUpdate is not None:
            super()._update_fromDict(
                fields_toUpdate, assert_keyInUpdateDict_isValid=True
            )

        if kwargs:
            super()._update_fromDict(kwargs, assert_keyInUpdateDict_isValid=False)
