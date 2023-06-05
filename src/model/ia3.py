import torch
import torch.nn as nn
import re
from src.model.utils import softmax_with_temperature

class IA3LinearLayer(nn.Module):
    def __init__(self, linear_layer, model_config=None, kwargs=None):
        super().__init__()
        self.linear_layer = linear_layer
        self.model_config = model_config
        self.temperature = self.model_config.temperature

        # Used when learning lambda parameters.
        if model_config and model_config.search_lambda == "learn":
            # checking if the tasks lambda is shared or not.
            # if it is shared then it must be present in kwargs and we will assign it
            # if not shared then we will create a mixing weight for each ia3 layer with the given initial value
            self.task_mixing_weights = kwargs["task_mixing_weights"]

            # Initialize the ia3 vectors for all the task plus the pretrained vectors.
            self.ia3_vector = nn.Parameter(
                torch.ones(
                    self.model_config.num_vectors,
                    self.linear_layer.out_features,
                )
            )

        # Used in cases where an IA3 vector is already provided.
        else:
            self.ia3_vector = nn.Parameter(
                torch.ones(
                    self.linear_layer.out_features,
                )
            )

    def forward(self, input):
        hidden = self.linear_layer(input)

        if self.model_config and self.model_config.search_lambda == "learn":
            assert hasattr(
                self, "task_mixing_weights"
            ), f"Task mixing weights not defined!"


            # multiply the mixing weights with the corresponding ia3 vectors and summing.
            # mixing_weight = self.self.task_mixing_weights.unsqueeze(-1)
            mixing_weight = softmax_with_temperature(self.task_mixing_weights, self.temperature, dim=-1).unsqueeze(-1)
            mixed_ia3_vector = (mixing_weight * self.ia3_vector).sum(0)
            assert mixed_ia3_vector.shape[-1] == self.linear_layer.out_features

            # TODO: find out what was the issue with the original code.
            # self.mixed_ia3_vecs = final_mixed_weights.matmul(self.ia3_vector)
            return hidden * mixed_ia3_vector
        else:
            return hidden * self.ia3_vector


def modify_withIA3(transformer, model_config=None):
    """

    Args:
        transformer:

    Returns:

    """
    module_toModify = ".*SelfAttention|.*EncDecAttention|.*DenseReluDense"
    children_toModify = "k|v|wi_1.*"
    trainableParameter_regex = ".*ia3.*"

    ia3_kwargs = {}
    if model_config and model_config.search_lambda == "learn":
        if model_config.lambda_type in ["scalar-shared", "vector-shared"]:

            transformer.task_mixing_weights = nn.Parameter(
                torch.ones(model_config.num_vectors) * model_config.lambda_init
            )
            ia3_kwargs["task_mixing_weights"] = transformer.task_mixing_weights

    for module_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(module_toModify, module_name):
            for children_name, children in dict(module.named_children()).items():
                if re.fullmatch(children_toModify, children_name):
                    assert isinstance(children, nn.Linear)
                    setattr(
                        module,
                        children_name,
                        IA3LinearLayer(
                            children,
                            model_config,
                            ia3_kwargs,
                        ),
                    )

    return transformer, trainableParameter_regex
