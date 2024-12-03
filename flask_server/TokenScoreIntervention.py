import torch
import torch.nn.functional as F
import numpy as np

# ToDo's:
# Implement IntervenedGenerationController
# save interventions in InterventionMethods, add setter-method and apply interventions on generation
# self.TOP_K
# add_hooks: use model_wrapper functionality
# remove_hooks: use model_wrapper functionality
# Implement usage of above to LM-Debugger
# Move Functionalities to own files
# Hopefully find no bugs

class InterventionGenerationController:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.intervention_methods = []

    def register_method(self, method):
        self.intervention_methods.append(method)

    def generate(self, prompt, generate_k):
        # Setup Intervention-Hooks
        for intervention_method in self.intervention_methods:
            intervention_method.setup_intervention_hooks(prompt)

        response_dict = {}
        tokens = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.model_wrapper.device)
        greedy_output = self.model_wrapper.model.generate(**tokens,
                                            max_length=generate_k + len(tokens['input_ids'][0]))
        greedy_output = self.model_wrapper.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        response_dict['generate_text'] = greedy_output

        self.model_wrapper.clear_hooks()

        return response_dict


class TokenScoreInterventionMethod:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

        self.interventions = []

        raise NotImplementedError("This class is an Interface")

    def add_intervention(self, intervention):
        self.interventions.append(intervention)

    def set_interventions(self, interventions):
        self.interventions = interventions

    def clear_interventions(self):
        self.interventions = []

    def get_token_scores(self, prompt, interventions):
        raise NotImplementedError("This class is an Interface")

    def setup_intervention_hooks(self, prompt, interventions):
        raise NotImplementedError("This class is an Interface")


class LMDebuggerIntervention(TokenScoreInterventionMethod):
    def __init__(self, model_wrapper):
        super().__init__(model_wrapper)

    def process_pred_dict(self, pred_df):
        pred_d = {}
        pred_d['prompt'] = pred_df['sent']
        pred_d['layers'] = []
        for layer_n in range(self.model_wrapper.model.config.num_hidden_layers):
            layer_d = {}
            layer_d['layer'] = layer_n
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
            hooks_lst = []
            maxs_dict = {l: self.get_new_max_coef(l, pred_dict_raw) for l in range(self.model_wrapper.model.config.num_hidden_layers)}
            for intervention in self.interventions:
                if intervention['coeff'] > 0:
                    new_max_val = maxs_dict[intervention['layer']]
                else:
                    new_max_val = 0
                hooks_lst.append(self.set_control_hooks_gpt2({intervention['layer']: [intervention['dim']], },
                                                             coef_value=new_max_val))
            pred_dict_new_raw = self.process_and_get_data(prompt)
            pred_dict_new = self.process_pred_dict(pred_dict_new_raw)
            response_dict['intervention'] = pred_dict_new
            for hook in hooks_lst:
                self.remove_hooks(hook)
        return response_dict

    """
    Generation-Hook-Setting
    """

    def set_control_hooks_gpt2(self, values_per_layer, coef_value=0):
        def change_values(values, coef_val):
            def hook(module, input, output):
                output[:, :, values] = coef_val

            return hook

        hooks = []
        for l in range(self.model_wrapper.model.config.num_hidden_layers):
            if l in values_per_layer:
                values = values_per_layer[l]
            else:
                values = []
            hook = self.model_wrapper.model.model.layers[l].mlp.gate_proj.register_forward_hook(
                change_values(values, coef_value)
            )
            hooks.append(hook)
            hook = self.model_wrapper.model.model.layers[l].mlp.up_proj.register_forward_hook(
                change_values(values, coef_value)
            )
            hooks.append(hook)

        return hooks

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

        self.model_wrapper.model.model.layers[0].input_layernorm.register_forward_hook(get_activation("input_embedding"))

        for i in range(self.model_wrapper.model.config.num_hidden_layers):
            if i != 0:
                self.model_wrapper.model.model.layers[i].input_layernorm.register_forward_hook(get_activation("layer_residual_" + str(i - 1)))
            self.model_wrapper.model.model.layers[i].post_attention_layernorm.register_forward_hook(get_activation("intermediate_residual_" + str(i)))

            self.model_wrapper.model.model.layers[i].self_attn.register_forward_hook(get_activation("attn_" + str(i)))
            self.model_wrapper.model.model.layers[i].mlp.register_forward_hook(get_activation("mlp_" + str(i)))
            self.model_wrapper.model.model.layers[i].mlp.down_proj.register_forward_hook(get_activation("m_coef_" + str(i)))

        self.model_wrapper.model.model.norm.register_forward_hook(get_activation("layer_residual_" + str(final_layer)))

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
            hooks_lst = []
            maxs_dict = {l: self.get_new_max_coef(l, pred_dict_raw) for l in range(self.model_wrapper.model.config.num_hidden_layers)}
            for intervention in self.interventions:
                if intervention['coeff'] > 0:
                    new_max_val = maxs_dict[intervention['layer']]
                else:
                    new_max_val = 0
                hooks_lst.append(self.set_control_hooks_gpt2({intervention['layer']: [intervention['dim']], },
                                                             coef_value=new_max_val))
