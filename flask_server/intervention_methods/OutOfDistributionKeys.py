from .InterventionMethod import MetricItem
from .EasyEdit.easyeditor.models.rome.compute_u import get_inv_cov, compute_u
from .EasyEdit.easyeditor.models.rome.rome_main import get_context_templates

class OutOfDistributionKeysMetric(MetricItem):
    def __init__(self, controller, args, layer):
        super().__init__(controller, args, layer=layer)

    def calculate_metric(self, token_logits):
        # Find ROME-Modules
        rome_modules = list(
            filter(
                lambda x: x.get_representation() == "ROME",
                self.controller.intervention_methods
            )
        )

        self.metric_value = {}

        for rome_module in rome_modules:
            hparams = rome_module.ee_hparams
            inv_cov = get_inv_cov(
                self.controller.model_wrapper.model,
                self.controller.model_wrapper.tokenizer,
                hparams.rewrite_module_tmp.format(rome_module.layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,
                hparams=hparams,
            )

            requests = [{
                "prompt": intervention["text_inputs"]["prompt"],
                "subject": intervention["text_inputs"]["subject"],
                "target_new": intervention["text_inputs"]["target"]
            } for intervention in rome_module.interventions if intervention["coeff"] > 0.0]

            for request in requests:
                k_vec = compute_u(
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

                k_vec_new_dt = k_vec.to(inv_cov.dtype)

                self.metric_value[f"L{rome_module.layer} | {request['subject']}"] = (k_vec_new_dt.T @ inv_cov @ k_vec_new_dt).item()

    def get_frontend_representation(self):
        return {
            "text_outputs": self.metric_value
        }