import torch
from .MetricItem import MetricItem


class PerplexityMetric(MetricItem):
    def __init__(self, controller):
        super().__init__(controller)

    def get_text_outputs(self, prompt, token_logits, additional_params=None):
        log_probs = torch.nn.functional.log_softmax(token_logits)
        max_log_probs = torch.max(log_probs, dim=1).values
        return {
            "Perplexity": torch.exp(-1 * max_log_probs.sum() / len(max_log_probs)).item()
        }
