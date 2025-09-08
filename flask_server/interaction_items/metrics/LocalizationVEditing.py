import torch
from .MetricItem import MetricItem

from .causal_trace.causal_trace import ModelAndTokenizer, calculate_hidden_flow


class LocalizationVEditingMetric(MetricItem):
    def __init__(self, controller):
        super().__init__(controller)

        # Request for additional Parameter 'interventions'
        self.parameters.need_parameter("interventions")

        self.model_and_tokenizer = ModelAndTokenizer(
            model=self.controller.model_wrapper.model,
            tokenizer=self.controller.model_wrapper.tokenizer
        )

    def pre_intervention_hook(self, prompt, additional_params=None):
        metric_values = {}

        for intervention in additional_params["interventions"]:
            subject = intervention["text_inputs"]["subject"]

            try:
                hidden_flow_rv = calculate_hidden_flow(
                    self.model_and_tokenizer,
                    prompt,
                    subject=subject,
                    samples=10,
                    noise=0.1,
                    window=10,
                    kind="mlp",
                    device=self.controller.model_wrapper.model.device
                )

                score_per_layer = torch.max(hidden_flow_rv["scores"], dim=0).values

                highest_scoring_layer = torch.argmax(score_per_layer).item()
                score = torch.max(score_per_layer).item()

                metric_values[subject] = f"L{highest_scoring_layer} | {score}"
            except ValueError:
                metric_values[subject] = "Invalid: Subject not in Prompt"

        self.parameters.parameters_retrieval_functions["metric_values"] = lambda: metric_values
        self.parameters.need_parameter("metric_values")

    def get_text_outputs(self, prompt, token_logits, additional_params=None):
        return additional_params["metric_values"]
