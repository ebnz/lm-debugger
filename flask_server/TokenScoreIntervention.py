import pickle

import torch
import torch.nn.functional as F
import numpy as np

from sparse_autoencoders import AutoEncoder

from create_offline_files import create_elastic_search_data, create_streamlit_data


# ToDo's:
# Move Functionalities to own files
# Hopefully find no bugs

class InterventionGenerationController:
    def __init__(self, model_wrapper, top_k):
        self.model_wrapper = model_wrapper
        self.TOP_K = top_k
        self.interventions = []
        self.intervention_methods = []

    def register_method(self, method):
        self.intervention_methods.append(method)

    def set_interventions(self, interventions):
        self.clear_interventions()

        self.interventions = interventions

        for intervention in self.interventions:
            intervention_type = intervention["type"]
            fitting_method_found = False
            for method in self.intervention_methods:
                if method.__class__.__name__ == intervention_type:
                    if intervention_type == "SAEIntervention" and intervention["layer"] != method.config["LAYER_INDEX"]:
                        continue
                    method.add_intervention(intervention)
                    fitting_method_found = True
            if not fitting_method_found:
                raise AttributeError(f"Intervention <{intervention}> has no fitting Intervention-Method!")


    def clear_interventions(self):
        self.interventions = []

        for method in self.intervention_methods:
            method.clear_interventions()

    def setup_intervention_hooks(self, prompt):
        for method in self.intervention_methods:
            method.setup_intervention_hooks(prompt)

    def generate(self, prompt, generate_k):
        # Setup Intervention-Hooks
        self.setup_intervention_hooks(prompt)

        response_dict = {}
        tokens = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.model_wrapper.device)
        greedy_output = self.model_wrapper.model.generate(**tokens,
                                            max_length=generate_k + len(tokens['input_ids'][0]))
        greedy_output = self.model_wrapper.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        response_dict['generate_text'] = greedy_output

        self.model_wrapper.clear_hooks()

        return response_dict

    def get_projections(self, type, layer, dim):
        for method in self.intervention_methods:
            if type != method.__class__.__name__:
                continue
            if layer not in method.supported_layers:
                continue
            rv = method.get_projections(layer=layer, dim=dim)
            return rv


class TokenScoreInterventionMethod:
    def __init__(self, model_wrapper, args, supported_layers):
        self.args = args
        self.model_wrapper = model_wrapper
        self.TOP_K = self.args.top_k_tokens_for_ui

        self.supported_layers = supported_layers

        self.interventions = []

    def add_intervention(self, intervention):
        self.interventions.append(intervention)

    def set_interventions(self, interventions):
        self.interventions = interventions

    def clear_interventions(self):
        self.interventions = []

    def get_token_scores(self, prompt):
        raise NotImplementedError(f"Intervention-Method <{self}> has no implemented <get_token_scores>")

    def setup_intervention_hooks(self, prompt):
        raise NotImplementedError(f"Intervention-Method <{self}> has no implemented <setup_intervention_hooks>")

    def get_projections(self, dim, *args, **kwargs):
        raise NotImplementedError(f"Intervention-Method <{self}> has no implemented <get_projections>")


class LMDebuggerIntervention(TokenScoreInterventionMethod):
    def __init__(self, model_wrapper, args):
        supported_layers = [i for i in range(model_wrapper.model.config_class().num_hidden_layers)]
        super().__init__(model_wrapper, args, supported_layers)

        self.dict_es = create_elastic_search_data(
            self.args.elastic_projections_path,
            self.model_wrapper.model,
            self.args.model_name,
            self.model_wrapper.tokenizer,
            self.args.top_k_for_elastic
        )

        if self.args.create_cluster_files:
            create_streamlit_data(
                self.args.streamlit_cluster_to_value_file_path,
                self.args.streamlit_value_to_cluster_file_path,
                self.model_wrapper.model,
                self.args.model_name,
                self.args.num_clusters
            )

    def process_pred_dict(self, pred_df):
        pred_d = {}
        pred_d['prompt'] = pred_df['sent']
        pred_d['layers'] = []
        for layer_n in range(self.model_wrapper.model.config.num_hidden_layers):
            layer_d = {}
            layer_d['layer'] = layer_n
            layer_d['type'] = self.__class__.__name__
            layer_d['predictions_before'] = [
                {'token': pred_df['residual_preds_tokens'][layer_n][k],
                 'score': float(pred_df['residual_preds_probs'][layer_n][k])
                 }
                for k in range(self.TOP_K)
            ]
            layer_d['predictions_after'] = [
                {'token': pred_df['layer_preds_tokens'][layer_n][k],
                 'score': float(pred_df['layer_preds_probs'][layer_n][k])
                 }
                for k in range(self.TOP_K)
            ]
            significant_values_lst = []
            dims_layer_n = pred_df['top_coef_idx'][layer_n]
            scores_layer_n = pred_df['top_coef_vals'][layer_n]
            for k in range(self.TOP_K):
                significant_values_lst.append(
                    {'layer': layer_n,
                     'dim': dims_layer_n[k],
                     'score': float(scores_layer_n[k])
                     }
                )
            layer_d['significant_values'] = significant_values_lst

            pred_d['layers'].append(layer_d)
        return pred_d

    def get_token_scores(self, prompt):
        response_dict = {}
        pred_dict_raw = self.process_and_get_data(prompt)
        pred_dict = self.process_pred_dict(pred_dict_raw)
        response_dict['response'] = pred_dict
        if len(self.interventions) > 0:
            maxs_dict = {l: self.get_new_max_coef(l, pred_dict_raw) for l in range(self.model_wrapper.model.config.num_hidden_layers)}
            for intervention in self.interventions:
                if intervention['coeff'] > 0:
                    new_max_val = maxs_dict[intervention['layer']]
                else:
                    new_max_val = 0
                self.set_control_hooks_gpt2({intervention['layer']: [intervention['dim']], }, coef_value=new_max_val)
            pred_dict_new_raw = self.process_and_get_data(prompt)
            pred_dict_new = self.process_pred_dict(pred_dict_new_raw)
            response_dict['intervention'] = pred_dict_new
            self.model_wrapper.clear_hooks()
        return response_dict

    """
    Generation-Hook-Setting
    """

    def set_control_hooks_gpt2(self, values_per_layer, coef_value=0):
        def change_values(values, coef_val):
            def hook(module, input, output):
                output[:, :, values] = torch.Tensor([coef_val]).to(self.model_wrapper.model.dtype)

            return hook

        for l in range(self.model_wrapper.model.config.num_hidden_layers):
            if l in values_per_layer:
                values = values_per_layer[l]
            else:
                values = []
            self.model_wrapper.setup_hook(
                change_values(values, coef_value),
                l,
                "mlp.gate_proj"
            )
            self.model_wrapper.setup_hook(
                change_values(values, coef_value),
                l,
                "mlp.up_proj"
            )

    def set_hooks_gpt2(self):
        final_layer = self.model_wrapper.model.config.num_hidden_layers - 1

        for attr in ["activations_"]:
            if not hasattr(self.model_wrapper.model, attr):
                setattr(self.model_wrapper.model, attr, {})

        def get_activation(name):
            def hook(module, input, output):
                if "mlp" in name or "attn" in name or "m_coef" in name:
                    if "attn" in name:
                        num_tokens = list(output[0].size())[1]  #(batch, sequence, hidden_state)
                        self.model_wrapper.model.activations_[name] = output[0][:, num_tokens - 1].detach()
                    elif "mlp" in name:
                        num_tokens = list(output[0].size())[0]  #(batch, sequence, hidden_state)
                        self.model_wrapper.model.activations_[name] = output[0][num_tokens - 1].detach()
                    elif "m_coef" in name:
                        num_tokens = list(input[0].size())[1]  #(batch, sequence, hidden_state)
                        self.model_wrapper.model.activations_[name] = input[0][:, num_tokens - 1].detach()
                elif "residual" in name or "embedding" in name:
                    num_tokens = list(input[0].size())[1]  #(batch, sequence, hidden_state)
                    if name == "layer_residual_" + str(final_layer):
                        self.model_wrapper.model.activations_[name] = self.model_wrapper.model.activations_[
                                                            "intermediate_residual_" + str(final_layer)] + \
                                                        self.model_wrapper.model.activations_["mlp_" + str(final_layer)]

                    else:
                        self.model_wrapper.model.activations_[name] = input[0][:,
                                                        num_tokens - 1].detach()

            return hook

        self.model_wrapper.setup_hook(
            get_activation("input_embedding"),
            0,
            "input_layernorm",
            permanent=False
        )

        for i in range(self.model_wrapper.model.config.num_hidden_layers):
            if i != 0:
                self.model_wrapper.setup_hook(
                    get_activation("layer_residual_" + str(i - 1)),
                    i,
                    "input_layernorm",
                    permanent=False
                )
            self.model_wrapper.setup_hook(
                get_activation("intermediate_residual_" + str(i)),
                i,
                "post_attention_layernorm",
                permanent=False
            )

            self.model_wrapper.setup_hook(
                get_activation("attn_" + str(i)),
                i,
                "self_attn",
                permanent=False
            )
            self.model_wrapper.setup_hook(
                get_activation("mlp_" + str(i)),
                i,
                "mlp",
                permanent=False
            )
            self.model_wrapper.setup_hook(
                get_activation("m_coef_" + str(i)),
                i,
                "mlp.down_proj",
                permanent=False
            )

        self.model_wrapper.setup_hook(
            get_activation("layer_residual_" + str(final_layer)),
            None,
            "norm",
            permanent=False
        )

    def get_resid_predictions(self, sentence, start_idx=None, end_idx=None, set_mlp_0=False):
        HIDDEN_SIZE = self.model_wrapper.model.config.hidden_size

        layer_residual_preds = []
        intermed_residual_preds = []

        if start_idx is not None and end_idx is not None:
            tokens = [
                token for token in sentence.split(' ')
                if token not in ['', '\n']
            ]

            sentence = " ".join(tokens[start_idx:end_idx])
        tokens = self.model_wrapper.tokenizer(sentence, return_tensors="pt")
        tokens.to(self.model_wrapper.device)
        output = self.model_wrapper.model(**tokens, output_hidden_states=True)

        for layer in self.model_wrapper.model.activations_.keys():
            if "layer_residual" in layer or "intermediate_residual" in layer:
                normed = self.model_wrapper.model.model.norm(self.model_wrapper.model.activations_[layer])

                logits = torch.matmul(self.model_wrapper.model.lm_head.weight, normed.T)

                probs = F.softmax(logits.T[0], dim=-1)

                probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

                assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs) - 1)) + layer

                probs_ = []
                for index, prob in enumerate(probs):
                    probs_.append((index, prob))
                top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:self.TOP_K]
                top_k = [(t[1].item(), self.model_wrapper.tokenizer.decode(t[0])) for t in top_k]
            if "layer_residual" in layer:
                layer_residual_preds.append(top_k)
            elif "intermediate_residual" in layer:
                intermed_residual_preds.append(top_k)

            for attr in ["layer_resid_preds", "intermed_residual_preds"]:
                if not hasattr(self.model_wrapper.model, attr):
                    setattr(self.model_wrapper.model, attr, [])

            self.model_wrapper.model.layer_resid_preds = layer_residual_preds
            self.model_wrapper.model.intermed_residual_preds = intermed_residual_preds

    def get_preds_and_hidden_states(self, prompt):
        self.set_hooks_gpt2()

        sent_to_preds = {}
        sent_to_hidden_states = {}
        sentence = prompt[:]
        self.get_resid_predictions(sentence)
        sent_to_preds["layer_resid_preds"] = self.model_wrapper.model.layer_resid_preds
        sent_to_preds["intermed_residual_preds"] = self.model_wrapper.model.intermed_residual_preds
        sent_to_hidden_states = self.model_wrapper.model.activations_.copy()

        return sent_to_hidden_states, sent_to_preds

    def process_and_get_data(self, prompt):
        sent_to_hidden_states, sent_to_preds = self.get_preds_and_hidden_states(prompt)
        records = []
        top_coef_idx = []
        top_coef_vals = []
        residual_preds_probs = []
        residual_preds_tokens = []
        layer_preds_probs = []
        layer_preds_tokens = []
        for LAYER in range(self.model_wrapper.model.config.num_hidden_layers):
            coefs_ = []
            m_coefs = sent_to_hidden_states["m_coef_" + str(LAYER)].squeeze(0).cpu().numpy()
            res_vec = sent_to_hidden_states["layer_residual_" + str(LAYER)].squeeze(0).cpu().numpy()
            value_norms = torch.linalg.norm(self.model_wrapper.model.model.layers[LAYER].mlp.down_proj.weight.data, dim=0).cpu()
            scaled_coefs = np.absolute(m_coefs) * value_norms.numpy()

            for index, prob in enumerate(scaled_coefs):
                coefs_.append((index, prob))

            top_values = sorted(coefs_, key=lambda x: x[1], reverse=True)[:self.TOP_K]
            c_idx, c_vals = zip(*top_values)
            top_coef_idx.append(c_idx)
            top_coef_vals.append(c_vals)
            residual_p_probs, residual_p_tokens = zip(*sent_to_preds['intermed_residual_preds'][LAYER])
            residual_preds_probs.append(residual_p_probs)
            residual_preds_tokens.append(residual_p_tokens)

            layer_p_probs, layer_p_tokens = zip(*sent_to_preds['layer_resid_preds'][LAYER])
            layer_preds_probs.append(layer_p_probs)
            layer_preds_tokens.append(layer_p_tokens)

        return {
            "sent": prompt,
            "top_coef_idx": top_coef_idx,
            "top_coef_vals": top_coef_vals,
            "residual_preds_probs": residual_preds_probs,
            "residual_preds_tokens": residual_preds_tokens,
            "layer_preds_probs": layer_preds_probs,
            "layer_preds_tokens": layer_preds_tokens,
            "layer_residual_vec": res_vec,
        }

    def get_new_max_coef(self, layer, old_dict, eps=10e-3):
        curr_max_val = old_dict['top_coef_vals'][layer][0]
        return curr_max_val + eps

    def setup_intervention_hooks(self, prompt):
        pred_dict_raw = self.process_and_get_data(prompt)
        if len(self.interventions) > 0:
            maxs_dict = {l: self.get_new_max_coef(l, pred_dict_raw) for l in range(self.model_wrapper.model.config.num_hidden_layers)}
            for intervention in self.interventions:
                if intervention['coeff'] > 0:
                    new_max_val = maxs_dict[intervention['layer']]
                else:
                    new_max_val = 0
                self.set_control_hooks_gpt2({intervention['layer']: [intervention['dim']], }, coef_value=new_max_val)

    def process_clean_token(self, token):
        return token

    def get_projections(self, dim, layer=None, *args, **kwargs):
        if layer is None:
            raise AttributeError(f"No layer provided. Parameter layer: <{layer}>")
        x = [(x[1], x[2]) for x in self.dict_es[(int(layer), int(dim))]]
        new_d = {'layer': int(layer), 'dim': int(dim)}
        top_k = [{'token': self.process_clean_token(x[i][0]), 'logit': float(x[i][1])} for i in range(len(x))]
        new_d['top_k'] = top_k
        return new_d


class SAEIntervention(TokenScoreInterventionMethod):
    def __init__(self, model_wrapper, args, config_path, device):
        self.config_path = config_path
        self.device = device

        with open(self.config_path, "rb") as f:
            self.config = pickle.load(f)

        supported_layers = [self.config["LAYER_INDEX"]]
        super().__init__(model_wrapper, args, supported_layers)

        self.autoencoder = AutoEncoder.load_model_from_config(self.config)
        self.autoencoder.to(self.device)

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

    # ToDo: Add type to return dict and outsource building of dict to Controller. Only return logits
    # ToDo: Check for correctness
    def get_projections(self, dim, *args, **kwargs):
        layer_id = self.config["LAYER_INDEX"]

        f = torch.zeros(self.autoencoder.m)
        f[dim] = 1

        x_hat = self.autoencoder.forward_decoder(f.to(self.device)).detach().cpu().to(dtype=torch.float16)

        llm_layer_output = self.model_wrapper.model.model.layers[layer_id].mlp.down_proj(x_hat.to(self.model_wrapper.device))

        #normed_x_hat = self.model_wrapper.model.model.norm(llm_layer_output)
        logits = self.model_wrapper.model.lm_head(llm_layer_output).detach().cpu()
        argsorted_logits = np.argsort(-1 * logits)[:self.args.top_k_for_elastic].tolist()

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


