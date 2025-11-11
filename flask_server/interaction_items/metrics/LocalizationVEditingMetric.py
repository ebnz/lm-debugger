import torch
from .MetricItem import MetricItem, Attributes

from .causal_trace.causal_trace import ModelAndTokenizer, calculate_hidden_flow


class LocalizationVEditingMetric(MetricItem):
    """
    Performs a Causal Trace for each Layer, a Model-Transform Intervention is defined.
    Displays the Layer, where the knowledge is stored, according to Causal Tracing
    """
    def __init__(self, controller):
        super().__init__(controller)

        # Request for additional Parameters
        self.parameters.need_parameter(Attributes.INTERVENTIONS)

        self.model_and_tokenizer = ModelAndTokenizer(
            model=self.model_wrapper.model,
            tokenizer=self.model_wrapper.tokenizer
        )

    def pre_intervention_hook(self, prompt, INTERVENTIONS=None):
        metric_values = {}

        for intervention in INTERVENTIONS:
            # Filter out LMDebuggerIntervention
            if intervention["type"] == "LMDebuggerIntervention":
                continue

            subject = intervention["text_inputs"]["subject"]

            try:
                hidden_flow_rv = calculate_hidden_flow(
                    self.model_and_tokenizer,
                    prompt,
                    subject=subject,
                    samples=self.config.samples,
                    noise=self.config.noise,
                    window=self.config.window,
                    kind=self.config.kind,
                    device=self.model_wrapper.model.device
                )

                score_per_layer = torch.max(hidden_flow_rv["scores"], dim=0).values

                highest_scoring_layer = torch.argmax(score_per_layer).item()
                score = torch.max(score_per_layer).item()

                metric_values[subject] = f"Layer {highest_scoring_layer} | {round(score, 3)}"
            except ValueError:
                metric_values[subject] = "Invalid: Subject not in Prompt"

        return metric_values

    def get_text_outputs(self, prompt, token_logits, pre_hook_rv=None, **kwargs):
        return pre_hook_rv
