import warnings
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_autoencoders.TransformerModels import CodeLlamaModel
from intervention_methods.InterventionGenerationController import InterventionGenerationController
from intervention_methods.LMDebuggerIntervention import LMDebuggerIntervention
from intervention_methods.SAEIntervention import SAEIntervention
from intervention_methods.ROMEIntervention import ROMEIntervention


warnings.filterwarnings('ignore')


class ModelingRequests():
    def __init__(self, args):
        self.args = args

        # Use AutoModelForCausalLM and AutoTokenizer for ROME-Support
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        self.model_wrapper = CodeLlamaModel(model, tokenizer=tokenizer, device=args.device)

        self.intervention_controller = InterventionGenerationController(self.model_wrapper)

        # Load LMDebuggerIntervention
        self.intervention_controller.register_method(LMDebuggerIntervention(
            self.model_wrapper,
            self.args
        ))

        # Load SAEs
        for config_path in tqdm(self.args.sae_paths, desc="Loading SAE-Instances"):
            self.intervention_controller.register_method(SAEIntervention(
                self.model_wrapper,
                self.args,
                config_path
            ))

        # Load ROME Instances
        for config_path in tqdm(self.args.rome_paths, desc="Loading ROME-Instances"):
            self.intervention_controller.register_method(ROMEIntervention(
                self.model_wrapper,
                self.args,
                config_path
            ))

    def request2response(self, req_json_dict):
        prompt = req_json_dict['prompt']
        interventions = req_json_dict['interventions']

        # Generate Response-Dict without Interventions
        self.intervention_controller.clear_interventions()
        response_dict = self.intervention_controller.get_token_scores(prompt)

        # Generate Response-Dict with Interventions
        self.intervention_controller.set_interventions(interventions)
        intervention_dict = self.intervention_controller.get_token_scores(prompt)
        self.intervention_controller.clear_interventions()

        return {
            'response': response_dict,
            'intervention': intervention_dict
        } if len(intervention_dict['layers']) != 0 else {'response': response_dict}

    def request2response_for_generation(self, req_json_dict):
        prompt = req_json_dict['prompt']
        interventions = req_json_dict['interventions']
        generate_k = req_json_dict['generate_k']

        # Set Interventions
        self.intervention_controller.set_interventions(interventions)

        # Generate
        response_dict = self.intervention_controller.generate(prompt, generate_k)

        return response_dict

    def get_projections(self, type, layer, dim):
        return self.intervention_controller.get_projections(type, int(layer), int(dim))
