import torch
from .InterventionMethod import InterventionMethod
from .EasyEdit.easyeditor.util.alg_dict import ALG_DICT

class EasyEditInterventionMethod(InterventionMethod):
    def __init__(self, controller, layer, ee_hparams):
        super().__init__(controller, layer)

        self.ee_hparams = ee_hparams
        self.invoke_method = ALG_DICT[self.ee_hparams.alg_name]

        # This if-Statement and its contents are copied from ROME (https://github.com/aip-hd-research/my-rome)
        # Specifically from the my-rome/notebooks/rome.ipynb-Notebook
        if self.model_wrapper.tokenizer.pad_token is None:
            self.model_wrapper.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model_wrapper.model.config.pad_token_id = self.model_wrapper.tokenizer.pad_token_id
            self.model_wrapper.model.model.padding_idx = self.model_wrapper.model.config.pad_token_id
            self.model_wrapper.model.generation_config.pad_token_id = self.model_wrapper.tokenizer.pad_token_id
            # potentially resize embedding and set padding idx
            new_embedding_size = max(len(self.model_wrapper.tokenizer.vocab),
                                     self.model_wrapper.model.config.vocab_size)
            new_embedding = torch.nn.Embedding(new_embedding_size, self.model_wrapper.model.config.hidden_size,
                                               self.model_wrapper.tokenizer.pad_token_id)
            old_embedding = self.model_wrapper.model.get_input_embeddings()
            new_embedding.to(old_embedding.weight.device, old_embedding.weight.dtype)
            new_embedding.weight.data[:self.model_wrapper.model.config.vocab_size] = old_embedding.weight.data
            self.model_wrapper.model.set_input_embeddings(new_embedding)

    def get_name(self):
        return self.ee_hparams.alg_name

    def transform_model(self, prompt):
        request = [{
            "prompt": intervention["text_inputs"]["prompt"],
            "subject": intervention["text_inputs"]["prompt"],
            "target_new": intervention["text_inputs"]["target"]
        } for intervention in self.interventions if intervention["coeff"] > 0.0]

        rv = self.invoke_method(
            self.model_wrapper.model,
            self.model_wrapper.tokenizer,
            request,
            self.ee_hparams,
            copy=False
        )

        if isinstance(rv, tuple):
            edited_model = rv[0]
        else:
            edited_model = rv

        self.model_wrapper.model = edited_model

    def get_text_inputs(self):
        return {
            "prompt": "",
            "subject": "",
            "target": ""
        }

    def get_projections(self, dim, *args, **kwargs):
        return super().get_projections(dim, *args, **kwargs)

    def setup_intervention_hooks(self, prompt):
        return super().setup_intervention_hooks(prompt)