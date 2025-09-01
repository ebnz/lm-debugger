import torch
import numpy as np
import pickle

from .InterventionMethod import InterventionMethod
from sparse_autoencoders.AutoEncoder import AutoEncoder


class SAEIntervention(InterventionMethod):
    def __init__(self, model_wrapper, config, config_path):
        """
        Represents the Intervention Method using Sparse Autoencoders.
        :type model_wrapper: sparse_autoencoders.TransformerModelWrapper
        :type config: pyhocon.config_tree.ConfigTree
        :param model_wrapper: Model Wrapper, the Intervention Method is applied to
        :param config: Configuration-Options from LM-Debugger++'s JSONNET-Config File
        """
        self.config_path = config_path
        self.device = config.autoencoder_device

        with open(self.config_path, "rb") as f:
            self.config = pickle.load(f)

        supported_layers = [self.config["LAYER_INDEX"]]
        super().__init__(model_wrapper, config, supported_layers)

        self.active_coeff = config.sae_active_coeff

        self.autoencoder = AutoEncoder.load_model_from_config(self.config)
        self.autoencoder = self.autoencoder.to(self.device)

    def get_token_scores(self, prompt):
        def get_hook(layer_type):
            def hook(module, input, output):
                global activation_vector
                activation_vector = output

            if layer_type in ["mlp_activations", "attn_sublayer", "mlp_sublayer"]:
                return hook
            else:
                raise AttributeError(f"layer_type <{layer_type}> unknown")

        self.setup_intervention_hooks(prompt)

        layer_id = self.config["LAYER_INDEX"]
        layer_type = self.config["LAYER_TYPE"]

        self.model_wrapper.setup_hook(
            get_hook(layer_type),
            self.args["layer_mappings"][layer_type].format(layer_id)
        )

        tokens = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.model_wrapper.device)
        self.model_wrapper.model(**tokens)

        # Get Activations of the only Batch and last Token
        f = self.autoencoder.forward_encoder(activation_vector[0, -1].to(self.device))

        # Replace NaN's and Inf's to zero
        f_refined = f.nan_to_num(0.0)
        f_refined[f_refined == float("Inf")] = 0

        # Extract Top-K Features with Index and Activation-Score
        top_k_object = f_refined.topk(self.TOP_K)
        top_features = top_k_object.indices.tolist()
        top_scores = top_k_object.values.tolist()

        # Assemble Response
        response_dict = {
            "layers": [
                {
                    "layer": layer_id,
                    "significant_values": [
                        {
                            "dim": feature_index,
                            "layer": layer_id,
                            "score": score
                        } for feature_index, score in zip(top_features, top_scores)
                    ],
                    "type": self.get_name()
                }
            ]
        }

        return response_dict

    def setup_intervention_hooks(self, prompt):
        def get_hook(feature_index, new_value, layer_type):
            def hook(module, input, output):
                activation_vector = output
                f = self.autoencoder.forward_encoder(activation_vector.to(self.device))
                f[::, ::, feature_index] = new_value
                x_hat = self.autoencoder.forward_decoder(f).to(self.model_wrapper.device, dtype=torch.float16)
                return x_hat

            if layer_type in ["mlp_activations", "attn_sublayer", "mlp_sublayer"]:
                return hook
            else:
                raise AttributeError(f"layer_type <{layer_type}> unknown")

        # Insert Interventions
        for intervention in self.interventions:
            feature_index = intervention["dim"]
            coeff = self.active_coeff if intervention["coeff"] > 0 else 0
            layer_id = self.config["LAYER_INDEX"]
            layer_type = self.config["LAYER_TYPE"]

            self.model_wrapper.setup_hook(
                get_hook(feature_index, coeff, layer_type),
                self.args["layer_mappings"][layer_type].format(layer_id)
            )

    def get_projections(self, dim, *args, **kwargs):
        layer_id = self.config["LAYER_INDEX"]
        layer_type = self.config["LAYER_TYPE"]

        # Compute AutoEncoder-Output and set given Feature to high value
        f = torch.zeros(self.autoencoder.m)
        f[dim] = self.active_coeff

        # Calculate the output of the Decoder of the AutoEncoder
        x_hat = self.autoencoder.forward_decoder(f.to(self.device)).detach().cpu().to(dtype=torch.float16)

        # Calculate the Output of the MLP-Block of the Model
        if layer_type == "mlp_activations":
            res_stream = self.model_wrapper.model.model.layers[layer_id].mlp.down_proj(
                x_hat.to(self.model_wrapper.device)
            )
        elif layer_type == "attn_sublayer":
            res_stream = x_hat.to(self.model_wrapper.device)
        elif layer_type == "mlp_sublayer":
            res_stream = x_hat.to(self.model_wrapper.device)
        else:
            raise AttributeError(f"layer_type <{layer_type}> unknown")

        # Calculate the Output-Logits of the Model and select those with highest probability
        normed_res_stream = self.model_wrapper.model.model.norm(res_stream)
        logits = self.model_wrapper.model.lm_head(normed_res_stream).detach().cpu()
        argsorted_logits = np.argsort(-1 * logits)[:self.args.top_k_for_elastic].tolist()

        # Logits and Tokens with highest probability
        output_logits = logits[argsorted_logits].tolist()
        output_tokens = [self.model_wrapper.tokenizer._convert_id_to_token(item) for item in argsorted_logits]

        # Build Response
        top_k = [{
            "logit": logit,
            "token": token
        } for logit, token in zip(output_logits, output_tokens)]

        return {
            "dim": dim,
            "layer": layer_id,
            "top_k": top_k
        }
