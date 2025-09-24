import torch
from .MetricItem import MetricItem, Attributes


class ExcessiveWeightDeltasMetric(MetricItem):
    def __init__(self, controller):
        super().__init__(controller)

        # Request for additional Parameters
        self.parameters.need_parameter(Attributes.WEIGHT_DELTAS)
        self.parameters.need_parameter(Attributes.MANIPULATED_LAYERS)

    def get_text_outputs(self, prompt, token_logits, pre_hook_rv=None, MANIPULATED_LAYERS=None, WEIGHT_DELTAS=None):
        metric_values = {}

        for layer in MANIPULATED_LAYERS:
            down_descriptor = self.controller.config.layer_mappings["mlp_down_proj"].format(layer) + ".weight"

            metric_values[f"Layer {layer}"] = torch.linalg.matrix_norm(WEIGHT_DELTAS[down_descriptor]).item()

        return metric_values
