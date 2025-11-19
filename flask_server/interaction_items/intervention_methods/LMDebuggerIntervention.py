import torch
import torch.nn.functional as F
import numpy as np

from .InterventionMethod import InterventionMethod


class LMDebuggerIntervention(InterventionMethod):
    """
    Shows strongly activating Features of the MLP and their activation value.
    Feature activations can then be manipulated by clicking the Down-Arrow and (de-)activating a Feature
    using the corresponding slider in the Interventions-Menu
    """
    def __init__(self, controller):
        """
        Represents the original Intervention Method of the LM-Debugger.
        :type controller: controller.InterventionGenerationController
        :param controller: Controller for Intervention and Generation of this instance of the app
        """
        layers = [i for i in range(controller.model_wrapper.model.config.num_hidden_layers)]
        super().__init__(controller, layers)

        self.TOP_K = self.controller.config.top_k_tokens_for_ui

    """
    Frontend Definitions
    """
    def get_text_inputs(self):
        return super().get_text_inputs()

    def get_frontend_items(self, layer, prompt, *args, **kwargs):
        return self.get_token_scores(prompt)

    def get_api_layers(self, prompt):
        rv = []
        # Split original LMDebuggerIntervention into Metrics and Intervention part
        for layer in self.get_frontend_items(None, prompt):
            # Intervention Part
            rv.append({
                "docstring": self.__doc__ if self.__doc__ is not None else "This Intervention Method lacks a Docstring.",
                "layer": layer["layer"],
                "name": self.get_name(),
                "type": self.get_type(),
                "changeable_layer": self.get_changeable_layer(),
                "significant_values": layer["significant_values"]
            })

            # Metric Part
            rv.append({
                "docstring": "Shows the Top-k Tokens before and after the MLP of this Layer",
                "layer": layer["layer"],
                "name": "TopKTokens",
                "type": "metric",
                "changeable_layer": self.get_changeable_layer(),
                "predictions_before": layer["predictions_before"],
                "predictions_after": layer["predictions_after"]
            })

        return rv

    """
    Intervention Handling
    """
    def transform_model(self, intervention):
        return super().transform_model(intervention)

    def setup_intervention_hook(self, intervention, prompt):
        pred_dict_raw = self.process_and_get_data(prompt)
        maxs_dict = {
            layer: self.get_new_max_coef(layer, pred_dict_raw)
            for layer in range(self.model_wrapper.model.config.num_hidden_layers)
        }

        if intervention['coeff'] > 0:
            new_max_val = maxs_dict[intervention['layer']]
        else:
            new_max_val = 0
        self.set_control_hooks_gpt2(
            {intervention['layer']: [intervention['dim']]},
            coef_value=new_max_val
        )

    """
    Intervention Functionality
    Code from LM-Debugger
    """
    def get_token_scores(self, prompt):
        # additional_requested_token_idxs contains Token-Idxs of targets of all Knowledge-Edits (e.g. ROME, MEMIT)
        # These are shown in The Before/After MLP Tokens
        additional_requested_token_idxs = []
        for intervention in self.controller.interventions:
            if "text_inputs" not in intervention.keys():
                continue
            if "target" not in intervention["text_inputs"].keys():
                continue

            if "gpt" in self.model_wrapper.model_name:
                token_id = self.model_wrapper.tokenizer(f' {intervention["text_inputs"]["target"]}')["input_ids"][0]
            elif "llama" in self.model_wrapper.model_name:
                token_id = self.model_wrapper.tokenizer(
                    intervention["text_inputs"]["target"],
                    add_special_tokens=False
                )["input_ids"][0]
            else:
                token_id = self.model_wrapper.tokenizer(intervention["text_inputs"]["target"])["input_ids"][0]

            if token_id not in additional_requested_token_idxs:
                additional_requested_token_idxs.append(token_id)

        pred_dict_raw = self.process_and_get_data(
            prompt,
            additional_requested_token_idxs=additional_requested_token_idxs
        )
        pred_dict = self.process_pred_dict(pred_dict_raw)

        if len(self.interventions) <= 0:
            return pred_dict["layers"]

        elif len(self.interventions) > 0:
            maxs_dict = {
                layer: self.get_new_max_coef(layer, pred_dict_raw)
                for layer in range(self.model_wrapper.model.config.num_hidden_layers)
            }
            for intervention in self.interventions:
                if intervention['coeff'] > 0:
                    new_max_val = maxs_dict[intervention['layer']]
                else:
                    new_max_val = 0
                self.set_control_hooks_gpt2({intervention['layer']: [intervention['dim']], }, coef_value=new_max_val)
            pred_dict_new_raw = self.process_and_get_data(
                prompt,
                additional_requested_token_idxs=additional_requested_token_idxs
            )
            pred_dict_new = self.process_pred_dict(pred_dict_new_raw)
            self.model_wrapper.clear_hooks()
            return pred_dict_new["layers"]

    def process_pred_dict(self, pred_df):
        pred_d = {}
        pred_d['prompt'] = pred_df['sent']
        pred_d['layers'] = []
        for layer_n in range(self.model_wrapper.model.config.num_hidden_layers):
            layer_d = {}
            layer_d['layer'] = layer_n
            layer_d['type'] = self.get_name()
            # Top-K Tokens
            layer_d['predictions_before'] = [{
                'token': pred_df['residual_preds_tokens'][layer_n][k],
                'score': float(pred_df['residual_preds_probs'][layer_n][k])
                } for k in range(self.TOP_K)]
            layer_d['predictions_after'] = [{
                'token': pred_df['layer_preds_tokens'][layer_n][k],
                'score': float(pred_df['layer_preds_probs'][layer_n][k])
                } for k in range(self.TOP_K)]

            # Additionally requested Tokens
            if len(pred_df['residual_preds_tokens_add']) > 0:
                for i in range(len(pred_df['residual_preds_tokens_add'][layer_n])):
                    layer_d['predictions_before'].append({
                        'token': pred_df['residual_preds_tokens_add'][layer_n][i],
                        'score': float(pred_df['residual_preds_probs_add'][layer_n][i]),
                        'rank': float(pred_df['residual_preds_ranks_add'][layer_n][i])
                    })
                for i in range(len(pred_df['layer_preds_tokens_add'][layer_n])):
                    layer_d['predictions_after'].append({
                        'token': pred_df['layer_preds_tokens_add'][layer_n][i],
                        'score': float(pred_df['layer_preds_probs_add'][layer_n][i]),
                        'rank': float(pred_df['layer_preds_ranks_add'][layer_n][i])
                    })

            significant_values_lst = []
            dims_layer_n = pred_df['top_coef_idx'][layer_n]
            scores_layer_n = pred_df['top_coef_vals'][layer_n]
            for k in range(self.TOP_K):
                significant_values_lst.append({
                    'layer': layer_n,
                    'dim': dims_layer_n[k],
                    'score': float(scores_layer_n[k])
                })
            layer_d['significant_values'] = significant_values_lst

            pred_d['layers'].append(layer_d)
        return pred_d

    def set_control_hooks_gpt2(self, values_per_layer, coef_value=0):
        def change_values(values, coef_val):
            def hook(module, input, output):
                output[:, :, values] = torch.Tensor([coef_val]).to(self.model_wrapper.model.dtype)

            return hook

        for layer in range(self.model_wrapper.model.config.num_hidden_layers):
            if layer in values_per_layer:
                values = values_per_layer[layer]
            else:
                values = []
            self.model_wrapper.setup_hook(
                change_values(values, coef_value),
                self.controller.config["layer_mappings"]["mlp_gate_proj"].format(layer)
            )
            self.model_wrapper.setup_hook(
                change_values(values, coef_value),
                self.controller.config["layer_mappings"]["mlp_up_proj"].format(layer)
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
                        num_tokens = list(output[0].size())[1]  # (batch, sequence, hidden_state)
                        self.model_wrapper.model.activations_[name] = output[0][:, num_tokens - 1].detach()
                    elif "mlp" in name:
                        num_tokens = list(output[0].size())[0]  # (batch, sequence, hidden_state)
                        self.model_wrapper.model.activations_[name] = output[0][num_tokens - 1].detach()
                    elif "m_coef" in name:
                        num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                        self.model_wrapper.model.activations_[name] = input[0][:, num_tokens - 1].detach()
                elif "residual" in name or "embedding" in name:
                    num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                    if name == "layer_residual_" + str(final_layer):
                        self.model_wrapper.model.activations_[name] = self.model_wrapper.model.activations_[
                                                            "intermediate_residual_" + str(final_layer)] + \
                                                        self.model_wrapper.model.activations_["mlp_" + str(final_layer)]

                    else:
                        self.model_wrapper.model.activations_[name] = input[0][:, num_tokens - 1].detach()

            return hook

        self.model_wrapper.setup_hook(
            get_activation("input_embedding"),
            self.controller.config["layer_mappings"]["decoder_input_layernorm"].format(0)
        )

        for i in range(self.model_wrapper.model.config.num_hidden_layers):
            if i != 0:
                self.model_wrapper.setup_hook(
                    get_activation("layer_residual_" + str(i - 1)),
                    self.controller.config["layer_mappings"]["decoder_input_layernorm"].format(i)
                )

            self.model_wrapper.setup_hook(
                get_activation("intermediate_residual_" + str(i)),
                self.controller.config["layer_mappings"]["decoder_post_attention_layernorm"].format(i)
            )

            self.model_wrapper.setup_hook(
                get_activation("attn_" + str(i)),
                self.controller.config["layer_mappings"]["attn_sublayer"].format(i)
            )
            self.model_wrapper.setup_hook(
                get_activation("mlp_" + str(i)),
                self.controller.config["layer_mappings"]["mlp_sublayer"].format(i)
            )
            self.model_wrapper.setup_hook(
                get_activation("m_coef_" + str(i)),
                self.controller.config["layer_mappings"]["mlp_down_proj"].format(i)
            )

        self.model_wrapper.setup_hook(
            get_activation("layer_residual_" + str(final_layer)),
            self.controller.config["layer_mappings"]["post_decoder_norm"]
        )

    def get_resid_predictions(self, sentence, start_idx=None, end_idx=None, additional_requested_token_idxs=None):
        """
        Generates Residual Predictions of the original LMDebugger Intervention Method
        This Method obtains Top-K Tokens, and additionally requested Token Ranks
        :type sentence: str
        :param sentence: Prompt, used to calculate the Features/Token-Scores
        :type additional_requested_token_idxs: list[int]|None
        :param additional_requested_token_idxs: Token Indices of extra requested Tokens
        """
        # Top-K (Token, Prob)-Pairs before/after for each Layer
        layer_residual_preds = []
        intermed_residual_preds = []

        # Additionally requested (Token, Prob)-Pairs before/after for each Layer
        layer_residual_preds_add = []
        intermed_residual_preds_add = []

        if start_idx is not None and end_idx is not None:
            tokens = [
                token for token in sentence.split(' ')
                if token not in ['', '\n']
            ]

            sentence = " ".join(tokens[start_idx:end_idx])
        tokens = self.model_wrapper.tokenizer(sentence, return_tensors="pt")
        tokens.to(self.model_wrapper.device)
        self.model_wrapper.model(**tokens, output_hidden_states=True)

        for layer in self.model_wrapper.model.activations_.keys():
            if "layer_residual" in layer or "intermediate_residual" in layer:
                post_decoder_norm_module = self.model_wrapper.modules_dict[
                    self.controller.config["layer_mappings"]["post_decoder_norm"]
                ]
                normed = post_decoder_norm_module(self.model_wrapper.model.activations_[layer])

                logits = torch.matmul(self.model_wrapper.model.lm_head.weight.data, normed.T)

                probs = F.softmax(logits.T[0], dim=-1)

                probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

                probs_ = []
                for index, prob in enumerate(probs):
                    probs_.append((index, prob))
                top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:self.TOP_K]
                top_k = [(t[1].item(), self.model_wrapper.tokenizer.decode(t[0])) for t in top_k]

                additional_token_logit_tuples = []

                if additional_requested_token_idxs is not None:
                    additional_token_logit_tuples = [(
                        probs[idx],
                        self.model_wrapper.tokenizer.decode(idx),
                        np.argsort(-probs).argsort()[idx]
                    ) for idx in additional_requested_token_idxs]

            if "layer_residual" in layer:
                layer_residual_preds.append(top_k)
                layer_residual_preds_add.append(additional_token_logit_tuples)
            elif "intermediate_residual" in layer:
                intermed_residual_preds.append(top_k)
                intermed_residual_preds_add.append(additional_token_logit_tuples)

            for attr in ["layer_resid_preds", "intermed_residual_preds"]:
                if not hasattr(self.model_wrapper.model, attr):
                    setattr(self.model_wrapper.model, attr, [])

            self.model_wrapper.model.layer_resid_preds = layer_residual_preds
            self.model_wrapper.model.layer_resid_preds_add = layer_residual_preds_add
            self.model_wrapper.model.intermed_residual_preds = intermed_residual_preds
            self.model_wrapper.model.intermed_residual_preds_add = intermed_residual_preds_add

    def get_preds_and_hidden_states(self, prompt, additional_requested_token_idxs=None):
        self.set_hooks_gpt2()

        sent_to_preds = {}
        sent_to_hidden_states = {}
        sentence = prompt[:]
        self.get_resid_predictions(sentence, additional_requested_token_idxs=additional_requested_token_idxs)
        sent_to_preds["layer_resid_preds"] = self.model_wrapper.model.layer_resid_preds
        sent_to_preds["intermed_residual_preds"] = self.model_wrapper.model.intermed_residual_preds
        sent_to_preds["layer_resid_preds_add"] = self.model_wrapper.model.layer_resid_preds_add
        sent_to_preds["intermed_residual_preds_add"] = self.model_wrapper.model.intermed_residual_preds_add
        sent_to_hidden_states = self.model_wrapper.model.activations_.copy()

        return sent_to_hidden_states, sent_to_preds

    def process_and_get_data(self, prompt, additional_requested_token_idxs=None):
        sent_to_hidden_states, sent_to_preds = self.get_preds_and_hidden_states(
            prompt,
            additional_requested_token_idxs=additional_requested_token_idxs
        )

        top_coef_idx = []
        top_coef_vals = []

        # Normal Top-K Tokens/Probs
        residual_preds_probs = []
        residual_preds_tokens = []
        layer_preds_probs = []
        layer_preds_tokens = []

        # Additionally requested Tokens/Probs
        residual_preds_probs_add = []
        residual_preds_tokens_add = []
        residual_preds_ranks_add = []
        layer_preds_probs_add = []
        layer_preds_tokens_add = []
        layer_preds_ranks_add = []
        for LAYER in range(self.model_wrapper.model.config.num_hidden_layers):
            coefs_ = []
            m_coefs = sent_to_hidden_states["m_coef_" + str(LAYER)].squeeze(0).cpu().numpy()
            res_vec = sent_to_hidden_states["layer_residual_" + str(LAYER)].squeeze(0).cpu().numpy()

            mlp_down_weight = self.model_wrapper.modules_dict[
                self.controller.config["layer_mappings"]["mlp_down_proj"].format(LAYER)
            ].weight.data

            value_norms = torch.linalg.norm(
                mlp_down_weight,
                dim=1 if "gpt2" in self.model_wrapper.model_name else 0
            ).cpu()
            scaled_coefs = np.absolute(m_coefs) * value_norms.numpy()

            for index, prob in enumerate(scaled_coefs):
                coefs_.append((index, prob))

            # Get Top-K Features from MLP
            top_values = sorted(coefs_, key=lambda x: x[1], reverse=True)[:self.TOP_K]
            c_idx, c_vals = zip(*top_values)
            top_coef_idx.append(c_idx)
            top_coef_vals.append(c_vals)

            # Get Top-K (Idx, Logit)-Pairs from intermediate residual distribution
            residual_p_probs, residual_p_tokens = zip(*sent_to_preds['intermed_residual_preds'][LAYER])
            residual_preds_probs.append(residual_p_probs)
            residual_preds_tokens.append(residual_p_tokens)

            # Get Top-K (Idx, Logit)-Pairs from intermediate residual distribution
            layer_p_probs, layer_p_tokens = zip(*sent_to_preds['layer_resid_preds'][LAYER])
            layer_preds_probs.append(layer_p_probs)
            layer_preds_tokens.append(layer_p_tokens)

            # Get additionally requested (Idx, Logit)-Pairs from intermediate residual distribution
            if len(sent_to_preds['intermed_residual_preds_add'][LAYER]) != 0:
                residual_p_probs_add, residual_p_tokens_add, residual_p_ranks_add = zip(
                    *sent_to_preds['intermed_residual_preds_add'][LAYER]
                )
                residual_preds_probs_add.append(residual_p_probs_add)
                residual_preds_tokens_add.append(residual_p_tokens_add)
                residual_preds_ranks_add.append(residual_p_ranks_add)

                layer_p_probs_add, layer_p_tokens_add, layer_p_ranks_add = zip(
                    *sent_to_preds['layer_resid_preds_add'][LAYER]
                )
                layer_preds_probs_add.append(layer_p_probs_add)
                layer_preds_tokens_add.append(layer_p_tokens_add)
                layer_preds_ranks_add.append(layer_p_ranks_add)

        return {
            "sent": prompt,
            "top_coef_idx": top_coef_idx,
            "top_coef_vals": top_coef_vals,
            "residual_preds_probs": residual_preds_probs,
            "residual_preds_tokens": residual_preds_tokens,
            "layer_preds_probs": layer_preds_probs,
            "layer_preds_tokens": layer_preds_tokens,
            "residual_preds_probs_add": residual_preds_probs_add,
            "residual_preds_tokens_add": residual_preds_tokens_add,
            "residual_preds_ranks_add": residual_preds_ranks_add,
            "layer_preds_probs_add": layer_preds_probs_add,
            "layer_preds_tokens_add": layer_preds_tokens_add,
            "layer_preds_ranks_add": layer_preds_ranks_add,
            "layer_residual_vec": res_vec
        }

    def get_new_max_coef(self, layer, old_dict, eps=10e-3):
        curr_max_val = old_dict['top_coef_vals'][layer][0]
        return curr_max_val + eps

    def process_clean_token(self, token):
        return token

    """
    Intervention Feature Details
    """
    def get_projections(self, dim, layer=None, *args, **kwargs):
        if layer is None:
            raise AttributeError(f"No layer provided. Parameter layer: <{layer}>")

        # Project Features with Embedding Matrix
        token_embedding_weights = self.model_wrapper.modules_dict[
            self.controller.config["layer_mappings"]["token_embedding"]
        ].weight.data
        mlp_down_weights = self.model_wrapper.modules_dict[
            self.controller.config["layer_mappings"]["mlp_down_proj"].format(layer)
        ].weight.data

        if "gpt2" in self.model_wrapper.model_name:
            feature_projections = torch.matmul(
                token_embedding_weights,
                mlp_down_weights.T
            )
        else:
            feature_projections = torch.matmul(
                token_embedding_weights,
                mlp_down_weights
            ).T

        # Obtain Logits for one Feature
        feature_logits = feature_projections[dim].detach().cpu()

        # Sort top-k Tokens for the specific Feature
        argsorted_logits = np.argsort(-1 * feature_logits)[:self.controller.config.top_k_for_elastic].tolist()

        output_logits = feature_logits[argsorted_logits].tolist()
        output_tokens = [self.model_wrapper.tokenizer._convert_id_to_token(item) for item in argsorted_logits]

        top_k = [{
            "logit": logit,
            "token": token
        } for logit, token in zip(output_logits, output_tokens)]

        return {
            "dim": dim,
            "layer": layer,
            "top_k": top_k
        }
