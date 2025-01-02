import torch

from .TokenScoreIntervention import TokenScoreInterventionMethod

# ROME
from .rome_files.rome import ROMEHyperParams
from .rome_files.rome import apply_rome_to_model
from .rome_files.util import nethook


class ROMEIntervention(TokenScoreInterventionMethod):
    def __init__(self, model_wrapper, args):
        self.rome_hparams = ROMEHyperParams.from_json(args.rome_hparams_path)
        super().__init__(model_wrapper, args, self.rome_hparams.layers)

        # This if-Statement and its contents are copied from ROME (https://github.com/aip-hd-research/my-rome)
        # Specifically from the my-rome/notebooks/rome.ipynb-Notebook
        if self.model_wrapper.tokenizer.pad_token is None:
            self.model_wrapper.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model_wrapper.model.config.pad_token_id = self.model_wrapper.tokenizer.pad_token_id
            self.model_wrapper.model.model.padding_idx = self.model_wrapper.model.config.pad_token_id
            self.model_wrapper.model.generation_config.pad_token_id = self.model_wrapper.tokenizer.pad_token_id
            # potentially resize embedding and set padding idx
            new_embedding_size = max(len(self.model_wrapper.tokenizer.vocab), self.model_wrapper.model.config.vocab_size)
            new_embedding = torch.nn.Embedding(new_embedding_size, self.model_wrapper.model.config.hidden_size, self.model_wrapper.tokenizer.pad_token_id)
            old_embedding = self.model_wrapper.model.get_input_embeddings()
            new_embedding.to(old_embedding.weight.device, old_embedding.weight.dtype)
            new_embedding.weight.data[:self.model_wrapper.model.config.vocab_size] = old_embedding.weight.data
            self.model_wrapper.model.set_input_embeddings(new_embedding)

    """
    This Function is adapted from the my-rome/notebooks/rome.ipynb-Notebook of https://github.com/aip-hd-research/my-rome
    """
    def transform_model(self):
        # Generate Request-Object for ROME-API
        requests = [
            {
                "prompt": intervention["prompt"],
                "subject": intervention["subject"],
                "target_new": {"str": intervention["target"]},
            } for intervention in self.interventions
        ]

        nethook.set_requires_grad(True, self.model_wrapper.model)

        # Retrieve ROME-transformed Model and replace old model
        model_new, self.orig_weights = apply_rome_to_model(
            self.model_wrapper.model, self.model_wrapper.tokenizer, requests, self.rome_hparams, return_orig_weights=True
        )

        self.model_wrapper.model = model_new

    """
    This Function is adapted from the my-rome/notebooks/rome.ipynb-Notebook of https://github.com/aip-hd-research/my-rome
    """
    def restore_original_model(self):
        if self.orig_weights is not None:
            with torch.no_grad():
                for k, v in self.orig_weights.items():
                    nethook.get_parameter(self.model_wrapper.model, k)[...] = v

    def get_token_scores(self, prompt):
        response_dict = {"response": {"layers": [
            {
                "layer": self.rome_hparams.layers,
                "text_inputs": {
                    "prompt": "",
                    "subject": "",
                    "target": ""
                },
                "type": self.__class__.__name__
            }
        ]}}

        return response_dict
