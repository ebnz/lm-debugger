import torch
from .MetricItem import MetricItem


class ExcessiveWeightDeltasMetric(MetricItem):
    def __init__(self, controller):
        super().__init__(controller)

        # Request for additional Parameter 'weight_deltas'
        self.parameters.need_parameter("weight_deltas")

    def get_text_outputs(self, prompt, token_logits, additional_params=None):
        metric_values = {}

        # Get all Layers that are manipulated by Intervention Methods
        manip_layers = self.controller.get_manipulated_layers()
        deltas = additional_params["weight_deltas"]

        for layer in manip_layers:
            down_descriptor = self.config.layer_mappings["mlp_down_proj"].format(layer) + ".weight"

            metric_values[f"L{layer}"] = torch.linalg.matrix_norm(deltas[down_descriptor]).item()

        return metric_values
