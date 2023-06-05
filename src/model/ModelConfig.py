from src.utils.Config import Config


class ModelConfig(Config):
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

        self.pretrained_model = "bigscience/T0_3B"
        self.max_seq_len = 512

        self.peft_method = None
        self.peft_params = None

        self.checkpoint_to_directly_load_model = None

        self.function_to_merge_checkpoints = None
        self.model_checkpoints = None
        self.trainable_param_regex = None

        self.model_lambda = 1
        self.search_lambda = "none"
        self.temperature = 1
        self.learn_pretraining_lambda = False
        self.pretraining_init = 1
        self.lambda_init = 0.2
        self.lambda_type = (
            "vector-shared"  # scalar-shared, vector-shared, vector-notshared
        )
        self.pretrained_lambda_type = "scalar-shared"  # scalar-shared, scalar-notshared

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
