import torch
from .InterventionMethod import MetricItem

class PerplexityMetric(MetricItem):
    def __init__(self, controller, args, layer):
        super().__init__(controller, args, layer=layer)

    def calculate_metric(self, token_logits):
        log_probs = torch.nn.functional.log_softmax(token_logits)
        max_log_probs = torch.max(log_probs, dim=1).values
        self.metric_value = torch.exp(-1 * max_log_probs.sum() / len(max_log_probs)).item()

    def get_frontend_representation(self):
        return {
            "text_outputs": {
                "Perplexity": self.metric_value
            }
        }