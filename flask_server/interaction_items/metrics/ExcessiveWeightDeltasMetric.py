import torch
from .MetricItem import MetricItem

class ExcessiveWeightDeltasMetric(MetricItem):
    def __init__(self, controller):
        super().__init__(controller)

    def get_text_outputs(self, token_logits):
        metric_values = {}

        # Get all Layers that are manipulated by Intervention Methods
        manip_layers = map(lambda x: x.layer, self.controller.intervention_methods)

        for layer in manip_layers:
            deltas = self.controller.get_weight_deltas(layer=layer)
            down_descriptor = self.config.layer_mappings["mlp_down_proj"].format(layer) + ".weight"

            metric_values[f"L{layer}"] = torch.linalg.matrix_norm(deltas[down_descriptor]).item()

        return metric_values
