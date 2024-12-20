import torch
import numpy as np
import pickle

from .TokenScoreIntervention import TokenScoreInterventionMethod
from sparse_autoencoders.AutoEncoder import AutoEncoder


class SAEIntervention(TokenScoreInterventionMethod):
    def __init__(self, model_wrapper, args, config_path, device):
        self.config_path = config_path
        self.device = device

        with open(self.config_path, "rb") as f:
            self.config = pickle.load(f)

        supported_layers = [self.config["LAYER_INDEX"]]
        super().__init__(model_wrapper, args, supported_layers)

        self.autoencoder = AutoEncoder.load_model_from_config(self.config)
        self.autoencoder = self.autoencoder.to(self.device)

    def get_token_scores(self, prompt):
        global activation_vector
        def get_hook(layer_type):
            # mlp_activations
            def hook_mlp_acts(module, input, output):
                global activation_vector
                activation_vector = output

            def hook_mlp_sublayer(module, input, output):
                global activation_vector
                activation_vector = output

            def hook_attn_sublayer(module, input, output):
                global activation_vector
                activation_vector = output

            if layer_type == "mlp_activations":
                return hook_mlp_acts
            elif layer_type == "attn_sublayer":
                return hook_attn_sublayer
            elif layer_type == "mlp_sublayer":
                return hook_mlp_sublayer
            else:
                raise AttributeError(f"layer_type <{layer_type}> unknown")

        self.setup_intervention_hooks(prompt)

        layer_id = self.config["LAYER_INDEX"]
        layer_type = self.config["LAYER_TYPE"]

        self.model_wrapper.setup_hook(
            get_hook(layer_type),
            layer_id,
            layer_type
        )

        tokens = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.model_wrapper.device)
        self.model_wrapper.model(**tokens)

        f = self.autoencoder.forward_encoder(activation_vector[0, -1].to(self.device))

        top_k_object = f.topk(self.TOP_K)
        top_features = top_k_object.indices.tolist()
        top_scores = top_k_object.values.tolist()

        response_dict = {"response": {"layers": [
            {
                "layer": layer_id,
                "significant_values": [
                    {
                        "dim": feature_index,
                        "layer": layer_id,
                        "score": score
                    } for feature_index, score in zip(top_features, top_scores)
                ],
                "type": self.__class__.__name__
            }
        ]}}

        return response_dict

    def setup_intervention_hooks(self, prompt):
        def get_hook(feature_index, new_value, layer_type):
            # mlp_activations
            def hook_mlp_acts(module, input, output):
                activation_vector = output
                f = self.autoencoder.forward_encoder(activation_vector.to(self.device))
                f[::, ::, feature_index] = new_value
                x_hat = self.autoencoder.forward_decoder(f).to(self.model_wrapper.device, dtype=torch.float16)
                return x_hat

            def hook_mlp_sublayer(module, input, output):
                activation_vector = output
                f = self.autoencoder.forward_encoder(activation_vector.to(self.device))
                f[::, ::, feature_index] = new_value
                x_hat = self.autoencoder.forward_decoder(f).to(self.model_wrapper.device, dtype=torch.float16)
                return x_hat

            def hook_attn_sublayer(module, input, output):
                activation_vector = output
                f = self.autoencoder.forward_encoder(activation_vector.to(self.device))
                f[::, ::, feature_index] = new_value
                x_hat = self.autoencoder.forward_decoder(f).to(self.model_wrapper.device, dtype=torch.float16)
                return x_hat

            if layer_type == "mlp_activations":
                return hook_mlp_acts
            elif layer_type == "attn_sublayer":
                return hook_attn_sublayer
            elif layer_type == "mlp_sublayer":
                return hook_mlp_sublayer
            else:
                raise AttributeError(f"layer_type <{layer_type}> unknown")

        for intervention in self.interventions:
            feature_index = intervention["dim"]
            coeff = intervention["coeff"]
            layer_id = self.config["LAYER_INDEX"]
            layer_type = self.config["LAYER_TYPE"]

            self.model_wrapper.setup_hook(
                get_hook(feature_index, coeff, layer_type),
                layer_id,
                layer_type
            )

    def get_projections(self, dim, *args, **kwargs):
        layer_id = self.config["LAYER_INDEX"]
        layer_type = self.config["LAYER_TYPE"]

        # Compute AutoEncoder-Output and set given Feature to high value
        f = torch.zeros(self.autoencoder.m)
        f[dim] = 100

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
        output_tokens = self.model_wrapper.tokenizer._convert_id_to_token(argsorted_logits)

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
