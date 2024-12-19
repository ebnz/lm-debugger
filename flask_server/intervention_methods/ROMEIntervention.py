import torch

from TokenScoreIntervention import TokenScoreInterventionMethod

# ROME
from ..rome.experiments.py.demo import load_alg

class ROMEIntervention(TokenScoreInterventionMethod):
    def __init__(self, model_wrapper, args, rome_hparams):
        super().__init__(model_wrapper, args, rome_hparams.layers)
        self.rome_hparams = rome_hparams

        self.RewritingParamsClass, self.apply_method, self.hparams_prefix, self.hparams_suffix = load_alg(
            "ROME" # ToDo: Add Algo-Name generically
        )

    def transform_model(self):
        # Generate Request-Object for ROME-API
        requests = [
            {
                "prompt": intervention.prompt,
                "subject": intervention.subject,
                "target_new": {"str": intervention.target},
            } for intervention in self.interventions
        ]

        # Retrieve ROME-transformed Model and replace old model
        self.model_new, self.orig_weights = self.apply_method(
            self.model_wrapper.model, self.model_wrapper.tokenizer, requests, self.rome_hparams, return_orig_weights=True
        )

        self.model_wrapper.model = self.model_new

    def restore_original_model(self):
        if self.orig_weights is not None:
            with torch.no_grad():
                for k, v in self.orig_weights.items():
                    from flask_server.rome.util import nethook
                    nethook.get_parameter(self.model_wrapper.model, k)[...] = v
