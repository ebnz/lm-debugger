import torch

from .MetricItem import MetricItem
from ..intervention_methods.EasyEdit.easyeditor.models.rome.compute_u import compute_u
from ..intervention_methods.EasyEdit.easyeditor.models.rome.rome_main import get_context_templates
from ..intervention_methods.EasyEdit.easyeditor.models.rome.compute_v import get_module_input_output_at_word


class ROMEUpdateDenominator(MetricItem):
    """
    Calculates the absolute value of the denominator of ROME’s Weight Delta Matrix.
    Exceptionally small values (<1) make the Weight Delta Matrix excessively large, leading to possible problems.
    The Pattern is: '<InterventionMethod> | <Layer> | <Prompt> --> <Denominator of Weight Delta Matrix>'
    """
    def __init__(self, controller):
        super().__init__(controller)

    def get_text_outputs(self, prompt, token_logits, pre_hook_rv=None, **kwargs):
        metric_values = {}

        for method in self.controller.intervention_methods:
            if method.get_name() not in ["ROME", "R-ROME"]:
                continue

            hparams = method.ee_hparams

            for intervention in method.interventions:
                prompt = intervention["text_inputs"]["prompt"]
                subject = intervention["text_inputs"]["subject"]
                target = intervention["text_inputs"]["target"]

                layer_idx = intervention["layer"]

                request = {
                    "prompt": prompt,
                    "subject": subject,
                    "target_new": target
                }

                context_templates = get_context_templates(
                        self.model_wrapper.model,
                        self.model_wrapper.tokenizer,
                        hparams.context_template_length_params
                    )

                inv_cov_times_key = compute_u(
                    self.model_wrapper.model,
                    self.model_wrapper.tokenizer,
                    request,
                    hparams,
                    layer_idx,
                    context_templates 
                )

                # There is a bug in ROME where one of the keys isn't properly prefixed
                # with the context_templates
                if method.get_name() == "ROME":
                    key, _ = get_module_input_output_at_word(
                        self.model_wrapper.model,
                        self.model_wrapper.tokenizer,
                        layer_idx,
                        context_template=request["prompt"],
                        word=request["subject"],
                        module_template=hparams.rewrite_module_tmp,
                        fact_token_strategy=hparams.fact_token
                    )

                # R-ROME fixes this bug
                elif method.get_name() == "R-ROME":
                    key = torch.stack([
                        get_module_input_output_at_word(
                            self.model_wrapper.model,
                            self.model_wrapper.tokenizer,
                            layer_idx,
                            context_template=template.format(request["prompt"]),
                            word=request["subject"],
                            module_template=hparams.rewrite_module_tmp,
                            fact_token_strategy=hparams.fact_token
                        )[0]
                        for template in context_templates
                    ]).mean(0)

                denominator_descriptor = (f'{method.get_name()} | '
                                          f'Layer {layer_idx} | '
                                          f'"{prompt.format(subject)} {target}"')
                metric_values[denominator_descriptor] = abs(torch.dot(key, inv_cov_times_key).item())

        return metric_values
