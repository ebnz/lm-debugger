import torch
from .InterventionMethod import MetricItem

class ExcessiveWeightDeltasMetric(MetricItem):
    def __init__(self, controller, args, layer):
        super().__init__(controller, args, layer=layer)

    def calculate_metric(self, token_logits):
        deltas = self.controller.get_weight_deltas(layer=5)
        down_descriptor = self.args.layer_mappings["mlp_down_proj"].format(self.layer) + ".weight"
        self.metric_value = torch.linalg.matrix_norm(deltas[down_descriptor]).item()

    def get_frontend_representation(self):
        return {
            "text_outputs": {
                "Frob. Score": self.metric_value
            }
        }
