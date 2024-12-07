import json
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from create_offline_files import create_elastic_search_data, create_streamlit_data
from transformers import LlamaForCausalLM, CodeLlamaTokenizer

from sparse_autoencoders.TransformerModels import CodeLlamaModel
from TokenScoreIntervention import InterventionGenerationController, LMDebuggerIntervention

warnings.filterwarnings('ignore')


class ModelingRequests():
    def __init__(self, args):
        self.args = args
        self.model_wrapper = CodeLlamaModel(args.model_name, device=args.device)
        self.dict_es = create_elastic_search_data(args.elastic_projections_path, self.model_wrapper.model, args.model_name,
                                                  self.model_wrapper.tokenizer, args.top_k_for_elastic)
        if args.create_cluster_files:
            create_streamlit_data(args.streamlit_cluster_to_value_file_path, args.streamlit_value_to_cluster_file_path,
                                  self.model_wrapper.model, args.model_name, args.num_clusters)

        self.intervention_controller = InterventionGenerationController(self.model_wrapper, args.top_k_tokens_for_ui)
        self.intervention_controller.register_method(LMDebuggerIntervention())

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
        prompt = req_json_dict['prompt']
        interventions = req_json_dict['interventions']

        # Set Interventions for LM-Debugger-Intervention
        self.intervention_controller.intervention_methods[0].set_interventions(interventions)

        # Generate Token-Scores
        response_dict = self.intervention_controller.intervention_methods[0].get_token_scores(prompt)

        return response_dict

    def request2response_for_generation(self, req_json_dict, save_json=False, res_json_path=None,
                                        res_json_intervention_path=None):
        prompt = req_json_dict['prompt']
        interventions = req_json_dict['interventions']
        generate_k = req_json_dict['generate_k']

        # Prepare Intervention-Methods for Generation
        # LM-Debugger-Intervention
        self.intervention_controller.intervention_methods[0].set_interventions(interventions)
        self.intervention_controller.intervention_methods[0].setup_intervention_hooks(prompt)

        # Sparse-AutoEncoders
        # ToDo: tbd

        # Generate
        response_dict = self.intervention_controller.generate(prompt, generate_k)

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
