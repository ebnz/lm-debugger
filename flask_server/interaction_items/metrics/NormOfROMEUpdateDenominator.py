import torch

from .MetricItem import MetricItem
from ..intervention_methods.EasyEdit.easyeditor.models.rome.compute_u import get_inv_cov, compute_u
from ..intervention_methods.EasyEdit.easyeditor.models.rome.rome_main import get_context_templates
from ..intervention_methods.EasyEdit.easyeditor.models.rome.compute_v import get_module_input_output_at_word


class NormOfROMEUpdateDenominator(MetricItem):
    """
    Calculates the value of the denominator of ROME’s Weight Delta Matrix.
    Exceptionally small values (<1) make the Weight Delta Matrix excessively large, leading to possible problems.
    The Pattern is: '<InterventionMethod> | <Layer> | <Prompt> --> <Denominator of Weight Delta Matrix>'
    """
    def __init__(self, controller):
        super().__init__(controller)

    def get_text_outputs(self, prompt, token_logits, pre_hook_rv=None, **kwargs):
        # Find ROME-Modules
        rome_modules = list(
            filter(
                lambda x: x.get_name() in self.config.applicable_intervention_methods,
                self.controller.intervention_methods
            )
        )

        metric_values = {}

        for rome_module in rome_modules:
            hparams = rome_module.ee_hparams

            for intervention in rome_module.interventions:
                prompt = intervention["text_inputs"]["prompt"]
                subject = intervention["text_inputs"]["subject"]
                target = intervention["text_inputs"]["target"]

                layer_idx = intervention["layer"]

                request = {
                    "prompt": prompt,
                    "subject": subject,
                    "target_new": target
                }

                left_vector = compute_u(
                    self.model_wrapper.model,
                    self.model_wrapper.tokenizer,
                    request,
                    hparams,
                    layer_idx,
                    get_context_templates(
                        self.model_wrapper.model,
                        self.model_wrapper.tokenizer,
                        hparams.context_template_length_params
                    )
                )

                cur_input, cur_output = get_module_input_output_at_word(
                    self.model_wrapper.model,
                    self.model_wrapper.tokenizer,
                    layer_idx,
                    context_template=request["prompt"],
                    word=request["subject"],
                    module_template=hparams.rewrite_module_tmp,
                    fact_token_strategy=hparams.fact_token
                )

                denominator_descriptor = (f'{rome_module.get_name()} | '
                                          f'Layer {layer_idx} | '
                                          f'"{prompt.format(subject)} {target}"')
                metric_values[denominator_descriptor] = abs(torch.dot(cur_input, left_vector).item())

        return metric_values
