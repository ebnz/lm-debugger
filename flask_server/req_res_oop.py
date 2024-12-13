import json
import warnings

from sparse_autoencoders.TransformerModels import CodeLlamaModel
from TokenScoreIntervention import InterventionGenerationController, LMDebuggerIntervention, SAEIntervention

warnings.filterwarnings('ignore')


class ModelingRequests():
    def __init__(self, args):
        self.args = args

        self.model_wrapper = CodeLlamaModel(args.model_name, device=args.device)

        self.intervention_controller = InterventionGenerationController(self.model_wrapper, args.top_k_tokens_for_ui)
        self.intervention_controller.register_method(LMDebuggerIntervention(self.model_wrapper, self.args))
        self.intervention_controller.register_method(SAEIntervention(
            self.model_wrapper,
            self.args,
            self.args.autoencoder_path,
            self.args.autoencoder_device
        ))

    def json_req_to_prompt_and_interventions_d(self, req_json_path):
        with open(req_json_path) as json_f:
            req = json.load(json_f)
        return [req['prompt']], req['interventions']

    def get_new_max_coef(self, layer, old_dict, eps=10e-3):
        curr_max_val = old_dict['top_coef_vals'][layer][0]
        return curr_max_val + eps

    def request2response(self, req_json_dict, save_json=False, res_json_path=None, res_json_intervention_path=None):
        prompt = req_json_dict['prompt']
        interventions = req_json_dict['interventions']

        # Set Interventions for LM-Debugger-Intervention
        self.intervention_controller.set_interventions(interventions)

        response_dict = {'prompt': prompt, 'layers': []}
        intervention_dict = {'prompt': prompt, 'layers': []}
        # Assemble Response-Dicts
        for method in self.intervention_controller.intervention_methods:
            # Generate Token-Scores
            rv = method.get_token_scores(prompt)
            response_dict['layers'] += rv['response']['layers']
            if 'intervention' in rv.keys():
                intervention_dict['layers'] += rv['intervention']['layers']

        self.model_wrapper.clear_hooks()

        return {
            'response': response_dict,
            'intervention': intervention_dict
        } if len(intervention_dict['layers']) != 0 else {'response': response_dict}

    def request2response_for_generation(self, req_json_dict, save_json=False, res_json_path=None,
                                        res_json_intervention_path=None):
        prompt = req_json_dict['prompt']
        interventions = req_json_dict['interventions']
        generate_k = req_json_dict['generate_k']

        # Set Interventions
        self.intervention_controller.set_interventions(interventions)

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

    def get_projections(self, type, layer, dim):
        return self.intervention_controller.get_projections(type, int(layer), int(dim))
