import json
import warnings
import os

import numpy as np
import torch
import torch.nn.functional as F
from create_offline_files import create_elastic_search_data, create_streamlit_data
from transformers import LlamaForCausalLM, CodeLlamaTokenizer

import pickle
from utils import AutoEncoder

warnings.filterwarnings('ignore')


class ModelingRequests():
    def __init__(self, args):
        self.args = args
        self.model = LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        self.llm_device = args.device
        self.model.to(self.llm_device)
        self.tokenizer = CodeLlamaTokenizer.from_pretrained(args.model_name)
        self.dict_es = create_elastic_search_data(args.elastic_projections_path, self.model, args.model_name,
                                                  self.tokenizer, args.top_k_for_elastic)
        if args.create_cluster_files:
            create_streamlit_data(args.streamlit_cluster_to_value_file_path, args.streamlit_value_to_cluster_file_path,
                                  self.model, args.model_name, args.num_clusters)
        self.TOP_K = args.top_k_tokens_for_ui

        #Sparse Coding
        print(" >>>> Loading Autoencoder")
        self.autoencoder_device = args.autoencoder_device
        self.autoencoder_config_files = os.listdir(args.autoencoder_inference_config_path)
        self.autoencoder_configs = []
        for config_file in self.autoencoder_config_files:
            with open(f"{args.autoencoder_inference_config_path}/{config_file}", "rb") as f:
                autoencoder_config: dict = pickle.load(f)
                self.autoencoder_configs.append(autoencoder_config)

        #Set Attributes for AutoEncoder-Loading-Procedure
        self.dict_vecs = None
        self.current_ae_index = None
        self.current_autoencoder_config = None
        self.autoencoder = None
        self.autoencoder_interpretations = []

        if len(self.autoencoder_configs) == 0:
            raise Exception("No AutoEncoders found!")
        self.activate_autoencoder(0)

    def set_control_hooks_gpt2(self, values_per_layer, coef_value=0):
        def change_values(values, coef_val):
            def hook(module, input, output):
                output[:, :, values] = coef_val
                return output

            return hook

        hooks = []
        for l in range(self.model.config.num_hidden_layers):
            if l in values_per_layer:
                values = values_per_layer[l]
            else:
                values = []

            hook = self.model.model.layers[l].mlp.gate_proj.register_forward_hook(  # ToDo: up_proj?
                change_values(values, coef_value)
            )
            hooks.append(hook)
            hook = self.model.model.layers[l].mlp.up_proj.register_forward_hook(  # ToDo: down_proj?
                change_values(values, coef_value)
            )
            hooks.append(hook)

        return hooks

    def set_control_hook_for_autoencoder(self, autoencoder_index, dim, coef_value, layer_type):
        """
        Set an Intervention-Forward-Hook with a specified AutoEncoder at a specific Dimension and Coefficient-Value
        :param autoencoder_index: Index of the used AutoEncoder
        :param dim: Index of the value in Dictionary-Vector to be manipulated
        :param coef_value: 1 for activate this Feature, 0 for deactivate this feature
        :param layer_type: "attn" for Attention-Sublayer or "mlp" for MLP-Sublayer
        """
        def change_values_mlp(dim, coef_val):
            def hook_mlp(module, input, output):
                # Activate AutoEncoder
                self.activate_autoencoder(autoencoder_index)

                # Encode
                X = output.to(self.autoencoder_device)
                f = self.autoencoder.forward_encoder(X).detach()

                # Patching
                f_patched = torch.zeros_like(f)
                f_patched[::, ::, dim] = coef_val

                # Decode and Add
                X_hat = X + self.autoencoder.forward_decoder(f_patched).to(output.dtype)

                # Return
                return X_hat.to(self.llm_device)

            return hook_mlp

        def change_values_attn(dim, coef_val):
            def hook_attn(module, input, output):
                #Activate AutoEncoder
                self.activate_autoencoder(autoencoder_index)

                #Encode
                X = output[0].to(self.autoencoder_device)
                f = self.autoencoder.forward_encoder(X).detach()

                #Patching
                f_patched = torch.zeros_like(f)
                f_patched[::, ::, dim] = coef_val

                #Decode and Add
                X_hat = X + self.autoencoder.forward_decoder(f_patched).to(output[0].dtype)

                #Return
                return (X_hat.to(self.llm_device), output[1], output[2])

            return hook_attn

        if layer_type == "mlp":
            layer_index = self.autoencoder_configs[autoencoder_index]["LAYER_INDEX"]

            hook = self.model.model.layers[layer_index].mlp.register_forward_hook(
                change_values_mlp(dim, coef_value)
            )
            return [hook]

        elif layer_type == "attn":
            layer_index = self.autoencoder_configs[autoencoder_index]["LAYER_INDEX"]

            hook = self.model.model.layers[layer_index].self_attn.register_forward_hook(
                change_values_attn(dim, coef_value)
            )
            return [hook]

        else:
            raise Exception(f"ERROR: Unknown layer_type: <{layer_type}>")

    def remove_hooks(self, hooks):
        for hook in hooks:
            hook.remove()

    def set_hooks_gpt2(self):
        final_layer = self.model.config.num_hidden_layers - 1

        for attr in ["activations_"]:
            if not hasattr(self.model, attr):
                setattr(self.model, attr, {})

        def get_activation(name):
            def hook(module, input, output):
                if "mlp" in name or "attn" in name or "m_coef" in name:
                    if "attn" in name:
                        num_tokens = list(output[0].size())[1]  #(batch, sequence, hidden_state)
                        self.model.activations_[name] = output[0][:, num_tokens - 1].detach()
                    elif "mlp" in name:
                        num_tokens = list(output[0].size())[0]  #(batch, sequence, hidden_state)
                        self.model.activations_[name] = output[0][num_tokens - 1].detach()
                    elif "m_coef" in name:
                        num_tokens = list(input[0].size())[1]  #(batch, sequence, hidden_state)
                        self.model.activations_[name] = input[0][:, num_tokens - 1].detach()
                elif "residual" in name or "embedding" in name:
                    num_tokens = list(input[0].size())[1]  #(batch, sequence, hidden_state)
                    if name == "layer_residual_" + str(final_layer):
                        self.model.activations_[name] = self.model.activations_[
                                                            "intermediate_residual_" + str(final_layer)] + \
                                                        self.model.activations_["mlp_" + str(final_layer)]

                    else:
                        self.model.activations_[name] = input[0][:,
                                                        num_tokens - 1].detach()

            return hook

        self.model.model.layers[0].input_layernorm.register_forward_hook(get_activation("input_embedding"))

        for i in range(self.model.config.num_hidden_layers):
            if i != 0:
                self.model.model.layers[i].input_layernorm.register_forward_hook(get_activation("layer_residual_" + str(i - 1)))
            self.model.model.layers[i].post_attention_layernorm.register_forward_hook(get_activation("intermediate_residual_" + str(i)))

            self.model.model.layers[i].self_attn.register_forward_hook(get_activation("attn_" + str(i)))
            self.model.model.layers[i].mlp.register_forward_hook(get_activation("mlp_" + str(i)))
            self.model.model.layers[i].mlp.down_proj.register_forward_hook(get_activation("m_coef_" + str(i)))

        self.model.model.norm.register_forward_hook(get_activation("layer_residual_" + str(final_layer)))

    def get_resid_predictions(self, sentence, start_idx=None, end_idx=None, set_mlp_0=False):
        HIDDEN_SIZE = self.model.config.hidden_size

        layer_residual_preds = []
        intermed_residual_preds = []

        if start_idx is not None and end_idx is not None:
            tokens = [
                token for token in sentence.split(' ')
                if token not in ['', '\n']
            ]

            sentence = " ".join(tokens[start_idx:end_idx])
        tokens = self.tokenizer(sentence, return_tensors="pt")
        tokens.to(self.llm_device)
        output = self.model(**tokens, output_hidden_states=True)

        for layer in self.model.activations_.keys():
            if "layer_residual" in layer or "intermediate_residual" in layer:
                normed = self.model.model.norm(self.model.activations_[layer])

                logits = torch.matmul(self.model.lm_head.weight, normed.T)

                probs = F.softmax(logits.T[0], dim=-1)

                probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

                assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs) - 1)) + layer

                probs_ = []
                for index, prob in enumerate(probs):
                    probs_.append((index, prob))
                top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:self.TOP_K]
                top_k = [(t[1].item(), self.tokenizer.decode(t[0])) for t in top_k]
            if "layer_residual" in layer:
                layer_residual_preds.append(top_k)
            elif "intermediate_residual" in layer:
                intermed_residual_preds.append(top_k)

            for attr in ["layer_resid_preds", "intermed_residual_preds"]:
                if not hasattr(self.model, attr):
                    setattr(self.model, attr, [])

            self.model.layer_resid_preds = layer_residual_preds
            self.model.intermed_residual_preds = intermed_residual_preds

    def get_preds_and_hidden_states(self, prompt):
        self.set_hooks_gpt2()

        sent_to_preds = {}
        sent_to_hidden_states = {}
        sentence = prompt[:]
        self.get_resid_predictions(sentence)
        sent_to_preds["layer_resid_preds"] = self.model.layer_resid_preds
        sent_to_preds["intermed_residual_preds"] = self.model.intermed_residual_preds
        sent_to_hidden_states = self.model.activations_.copy()

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
        for LAYER in range(self.model.config.num_hidden_layers):
            coefs_ = []
            m_coefs = sent_to_hidden_states["m_coef_" + str(LAYER)].squeeze(0).cpu().numpy()
            res_vec = sent_to_hidden_states["layer_residual_" + str(LAYER)].squeeze(0).cpu().numpy()
            value_norms = torch.linalg.norm(self.model.model.layers[LAYER].mlp.down_proj.weight.data, dim=0).cpu()
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

    def process_pred_dict(self, pred_df):
        pred_d = {}
        pred_d['prompt'] = pred_df['sent']
        pred_d['layers'] = []
        for layer_n in range(self.model.config.num_hidden_layers):
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

    def json_req_to_prompt_and_interventions_d(self, req_json_path):
        with open(req_json_path) as json_f:
            req = json.load(json_f)
        return [req['prompt']], req['interventions']

    def process_clean_token(self, token):
        return token

    def get_new_max_coef(self, layer, old_dict, eps=10e-3):
        curr_max_val = old_dict['top_coef_vals'][layer][0]
        return curr_max_val + eps

    def request2response(self, req_json_dict, save_json=False, res_json_path=None, res_json_intervention_path=None):
        response_dict = {}
        prompt, interventions_lst = req_json_dict['prompt'], req_json_dict['interventions']
        pred_dict_raw = self.process_and_get_data(prompt)
        pred_dict = self.process_pred_dict(pred_dict_raw)
        response_dict['response'] = pred_dict
        if len(interventions_lst) > 0:
            hooks_lst = []
            maxs_dict = {l: self.get_new_max_coef(l, pred_dict_raw) for l in range(self.model.config.num_hidden_layers)}
            for intervention in interventions_lst:
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

    def request2response_for_generation(self, req_json_dict, save_json=False, res_json_path=None,
                                        res_json_intervention_path=None):
        response_dict = {}
        prompt, interventions_lst = req_json_dict['prompt'], req_json_dict['interventions']
        pred_dict_raw = self.process_and_get_data(prompt)
        if len(interventions_lst) > 0:
            hooks_lst = []
            maxs_dict = {l: self.get_new_max_coef(l, pred_dict_raw) for l in range(self.model.config.num_hidden_layers)}
            for intervention in interventions_lst:
                if intervention['coeff'] > 0:
                    new_max_val = maxs_dict[intervention['layer']]
                else:
                    new_max_val = 0
                hooks_lst.append(self.set_control_hooks_gpt2({intervention['layer']: [intervention['dim']], },
                                                             coef_value=new_max_val))
        tokens = self.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.llm_device)
        greedy_output = self.model.generate(**tokens,
                                            max_length=req_json_dict['generate_k'] + len(tokens['input_ids'][0]))
        greedy_output = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        response_dict['generate_text'] = greedy_output
        if len(interventions_lst) > 0:
            for hook in hooks_lst:
                self.remove_hooks(hook)
        return response_dict

    def send_request_get_response(self, request_json_dict):
        return self.request2response(request_json_dict,
                                     save_json=False,
                                     res_json_path=None,
                                     res_json_intervention_path=None)

    def send_request_get_response_for_generation(self, request_json_dict):
        return self.request2response_for_generation(request_json_dict,
                                                    save_json=False,
                                                    res_json_path=None,
                                                    res_json_intervention_path=None)

    def get_projections(self, layer, dim):
        x = [(x[1], x[2]) for x in self.dict_es[(int(layer), int(dim))]]
        new_d = {'layer': int(layer), 'dim': int(dim)}
        top_k = [{'token': self.process_clean_token(x[i][0]), 'logit': float(x[i][1])} for i in range(len(x))]
        new_d['top_k'] = top_k
        return new_d

    #Sparse Coding
    def get_autoencoder_files(self):
        return self.autoencoder_config_files

    def activate_autoencoder(self, index):
        #If index out of range, return False
        if index < 0 or index >= len(self.autoencoder_configs):
            return False
        #If index doesn't change, don't reload AutoEncoder
        if self.current_ae_index == index:
            return True
        self.current_ae_index = index
        self.current_autoencoder_config = self.autoencoder_configs[index]
        self.autoencoder = AutoEncoder.load_model_from_config(self.current_autoencoder_config)
        self.autoencoder = self.autoencoder.to(self.autoencoder_device)
        self.autoencoder_interpretations = self.current_autoencoder_config["INTERPRETATIONS"]
        return True

    def get_max_autoencoder_neuron_per_token(self, prompt):
        def attn_hook(module, input, output):
            activations = output[0].detach().cpu()
            activations = activations.to(self.autoencoder_device)
            X_hat, f = self.autoencoder(activations.to(torch.float32))
            self.dict_vecs = f.detach().cpu()[0]  #self.dict_vecs is of shape [NUM_TOKENS, DICT_VEC_SIZE]

        forward_hook = self.model.model.layers[1].self_attn.register_forward_hook(attn_hook)

        tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        tokens.to(self.llm_device)
        output = self.model(**tokens, output_hidden_states=True)

        forward_hook.remove()

        if self.dict_vecs is None:
            print("WARN: No Dictionary-Vectors!")
            return {}

        output_dict = {
            "tokens_as_string": [],
            "token_ids": [],
            "neuron_ids": [],
            "interpretations": [],
            "neuron_activations": []
        }

        neuron_max_object = torch.max(self.dict_vecs, dim=1)
        neuron_ids = neuron_max_object.indices.tolist()
        neuron_activations = neuron_max_object.values.tolist()

        token_ids = tokens["input_ids"][0].tolist()
        tokens_as_string = self.tokenizer.convert_ids_to_tokens(token_ids)

        if len(neuron_ids) != len(token_ids) or len(token_ids) != len(tokens_as_string) or len(token_ids) != len(
                neuron_activations):
            print("WARN: Wrong number of neuron_ids, token_ids or tokens_as_string!")
            return {}
        for i in range(len(neuron_ids)):
            interpretation = self.get_neuron_interpretation(neuron_ids[i])

            output_dict["tokens_as_string"].append(tokens_as_string[i])
            output_dict["token_ids"].append(token_ids[i])
            output_dict["neuron_ids"].append(neuron_ids[i])
            output_dict["interpretations"].append(interpretation)

            neuron_act = neuron_activations[i] - self.current_autoencoder_config["MINS"][i]
            if self.current_autoencoder_config["MAXS"][i] == 0:
                neuron_act = 0
            else:
                neuron_act = neuron_act / self.current_autoencoder_config["MAXS"][i]

            output_dict["neuron_activations"].append(neuron_act)

        return output_dict

    def get_neuron_interpretation(self, neuron_id):
        if neuron_id not in self.autoencoder_interpretations:
            return ""
        return self.autoencoder_interpretations[neuron_id]

    def request2response_get_max_act_neurons(self, req_json_dict):
        prompt = req_json_dict['prompt']
        output_dict = self.get_max_autoencoder_neuron_per_token(prompt)

        response_dict = {
            "autoencoder_name": self.autoencoder_config_files[self.current_ae_index],
            "autoencoder_layer_type": self.current_autoencoder_config["LAYER_TYPE"],
            "autoencoder_layer_index": self.current_autoencoder_config["LAYER_INDEX"]
        }
        for item in output_dict.keys():
            response_dict[item] = output_dict[item]

        return response_dict

    def get_neuron_activation_per_token(self, prompt, neuron_id):
        neuron_id = int(neuron_id)

        def attn_hook(module, input, output):
            activations = output[0].detach().cpu()
            activations = activations.to(self.autoencoder_device)
            X_hat, f = self.autoencoder(activations.to(torch.float32))
            self.dict_vecs = f.detach().cpu()[0]  #self.dict_vecs is of shape [NUM_TOKENS, DICT_VEC_SIZE]

        forward_hook = self.model.model.layers[1].self_attn.register_forward_hook(attn_hook)

        tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        tokens.to(self.llm_device)
        output = self.model(**tokens, output_hidden_states=True)

        forward_hook.remove()

        if self.dict_vecs is None:
            print("WARN: No Dictionary-Vectors!")
            return {}

        output_dict = {
            "tokens_as_string": [],
            "token_ids": [],
            "neuron_ids": [],
            "interpretations": [],
            "neuron_activations": []
        }

        neuron_activations = self.dict_vecs[::, neuron_id].squeeze().tolist()

        token_ids = tokens["input_ids"][0].tolist()
        tokens_as_string = self.tokenizer.convert_ids_to_tokens(token_ids)

        if len(token_ids) != len(tokens_as_string) or len(token_ids) != len(neuron_activations):
            print("WARN: Wrong number of neuron_ids, token_ids or tokens_as_string!")
            return {}
        for i in range(len(token_ids)):
            interpretation = self.get_neuron_interpretation(neuron_id)

            output_dict["tokens_as_string"].append(tokens_as_string[i])
            output_dict["token_ids"].append(token_ids[i])
            output_dict["neuron_ids"].append(neuron_id)
            output_dict["interpretations"].append(interpretation)

            neuron_act = neuron_activations[i] - self.current_autoencoder_config["MINS"][neuron_id]
            if self.current_autoencoder_config["MAXS"][neuron_id] == 0:
                neuron_act = 0
            else:
                neuron_act = 10 * neuron_act / self.current_autoencoder_config["MAXS"][neuron_id]

            output_dict["neuron_activations"].append(neuron_act)

        return output_dict

    def request2resonse_get_neuron_act(self, req_json_dict):
        prompt = req_json_dict['prompt']
        neuron_id = req_json_dict['neuron_id']
        output_dict = self.get_neuron_activation_per_token(prompt, neuron_id)

        response_dict = {
            "autoencoder_name": self.autoencoder_config_files[self.current_ae_index],
            "autoencoder_layer_type": self.current_autoencoder_config["LAYER_TYPE"],
            "autoencoder_layer_index": self.current_autoencoder_config["LAYER_INDEX"]
        }
        for item in output_dict.keys():
            response_dict[item] = output_dict[item]

        return response_dict

    #Combination of Intervention-Methods
    """
    Combine the Intervention-Method from LM-Debugger and Sparse Autoencoders
    """
    def request2response_generate_intervened(self, req_json_dict):
        response_dict = {}
        prompt, interventions_lst, sae_interventions_lst = req_json_dict['prompt'], req_json_dict['interventions'], req_json_dict['sae_interventions']
        pred_dict_raw = self.process_and_get_data(prompt)

        hooks_lst = []
        if len(interventions_lst) > 0:
            maxs_dict = {l: self.get_new_max_coef(l, pred_dict_raw) for l in range(self.model.config.num_hidden_layers)}
            for intervention in interventions_lst:
                if intervention['coeff'] > 0:
                    new_max_val = maxs_dict[intervention['layer']]
                else:
                    new_max_val = 0
                hooks_lst.append(self.set_control_hooks_gpt2({intervention['layer']: [intervention['dim']], },
                                                             coef_value=new_max_val))

        if len(sae_interventions_lst) > 0:
            for intervention in sae_interventions_lst:
                if intervention['coeff'] == 0:
                    new_max_val = 0 #Set to MINS[feature]
                else:
                    new_max_val = 100 #Set to MAXS[feature]
                #Set the layer_type to 'attn' or 'mlp' for SAE-Intervention instead of 'up_down' for LM-Debugger-Intervention
                autoencoder_index = int(intervention["autoencoder_index"])
                dim = intervention["dim"]
                layer_type = self.autoencoder_configs[autoencoder_index]["LAYER_TYPE"]
                hooks_lst.append(self.set_control_hook_for_autoencoder(autoencoder_index,
                                                                       dim,
                                                                       intervention["coeff"],   #ToDo: Set to new_max_val and MINS/MAXS
                                                                       layer_type))

        tokens = self.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.llm_device)
        try:
            greedy_output = self.model.generate(**tokens,
                                                max_length=req_json_dict['generate_k'] + len(tokens['input_ids'][0]))
            greedy_output = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
            response_dict['generate_text'] = greedy_output
        except Exception as e:
            print(e)


        if len(interventions_lst) > 0 or len(sae_interventions_lst) > 0:
            for hook in hooks_lst:
                self.remove_hooks(hook)
        return response_dict
