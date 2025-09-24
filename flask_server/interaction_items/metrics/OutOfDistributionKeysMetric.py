import torch

from .MetricItem import MetricItem
from ..intervention_methods.EasyEdit.easyeditor.models.rome.compute_u import get_inv_cov, compute_u
from ..intervention_methods.EasyEdit.easyeditor.models.rome.rome_main import get_context_templates
from ..intervention_methods.EasyEdit.easyeditor.models.rome.compute_v import get_module_input_output_at_word


class OutOfDistributionKeysMetric(MetricItem):
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

            requests = [{
                "prompt": intervention["text_inputs"]["prompt"],
                "subject": intervention["text_inputs"]["subject"],
                "target_new": intervention["text_inputs"]["target"]
            } for intervention in rome_module.interventions]

            for request in requests:
                left_vector = compute_u(
                    self.controller.model_wrapper.model,
                    self.controller.model_wrapper.tokenizer,
                    request,
                    hparams,
                    rome_module.layer,
                    get_context_templates(
                        self.controller.model_wrapper.model,
                        self.controller.model_wrapper.tokenizer,
                        hparams.context_template_length_params
                    )
                )

                cur_input, _ = get_module_input_output_at_word(
                    self.controller.model_wrapper.model,
                    self.controller.model_wrapper.tokenizer,
                    rome_module.layer,
                    context_template=request["prompt"],
                    word=request["subject"],
                    module_template=hparams.rewrite_module_tmp,
                    fact_token_strategy=hparams.fact_token,
                )

                subject = request['subject']
                datapoint_name = f'Layer {rome_module.layer} | Subject "{subject}"'
                metric_values[datapoint_name] = torch.dot(cur_input, left_vector).item()

        return metric_values
