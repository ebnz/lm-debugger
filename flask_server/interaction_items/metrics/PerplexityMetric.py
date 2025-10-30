import torch
from .MetricItem import MetricItem


class PerplexityMetric(MetricItem):
    """
    Calculates the Perplexity of the Model Output.
    Perplexity is a measure of uncertainty of the LLM.
    High values correspond to high uncertainty.
    """
    def __init__(self, controller):
        super().__init__(controller)

    def get_text_outputs(self, prompt, token_logits, pre_hook_rv=None, **kwargs):
        log_probs = torch.nn.functional.log_softmax(token_logits)
        max_log_probs = torch.max(log_probs, dim=1).values
        return {
            "Perplexity": torch.exp(-1 * max_log_probs.sum() / len(max_log_probs)).item()
        }
