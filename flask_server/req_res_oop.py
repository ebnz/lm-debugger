import os
import warnings

import yaml
from tqdm import tqdm

# Controller
from .controller.InterventionGenerationController import InterventionGenerationController
from .controller.TransformerModels import TransformerModelWrapper

# Metrics
from .interaction_items.metrics.NormOfWeightUpdate import NormOfWeightUpdate
from .interaction_items.metrics.Perplexity import Perplexity
from .interaction_items.metrics.DenominatorOfROMEUpdate import DenominatorOfROMEUpdate
from .interaction_items.metrics.CausalTrace import CausalTrace
from .interaction_items.metrics.Efficacy import Efficacy

# Intervention Methods
from .interaction_items.intervention_methods.LMDebuggerIntervention import LMDebuggerIntervention
from .interaction_items.intervention_methods.EasyEditInterventionMethod import EasyEditInterventionMethod
from .interaction_items.intervention_methods.EasyEdit.easyeditor import (
    KNHyperParams,
    ROMEHyperParams,
    MEMITHyperParams,
    PMETHyperParams,
    DINMHyperParams,
    R_ROMEHyperParams,
    EMMETHyperParams,
    MENDHyperParams
)

warnings.filterwarnings('ignore')

hparams_mapping = {
    "KN": KNHyperParams,
    "ROME": ROMEHyperParams,
    "MEMIT": MEMITHyperParams,
    "PMET": PMETHyperParams,
    "DINM": DINMHyperParams,
    "R-ROME": R_ROMEHyperParams,
    "EMMET": EMMETHyperParams
}


class ModelingRequests:
    def __init__(self, config):
        self.config = config

        self.model_wrapper = TransformerModelWrapper(config.model_name, device=config.device)

        self.intervention_controller = InterventionGenerationController(self.model_wrapper, self.config)

        self.intervention_controller.register_method(LMDebuggerIntervention(
            self.intervention_controller
        ))

        for file_name in tqdm(os.listdir(config.easy_edit_hparams_path), desc="Loading EasyEdit Methods"):
            path_to_conf = os.path.join(config.easy_edit_hparams_path, file_name)
            with open(path_to_conf) as file_desc:
                method_config = yaml.safe_load(file_desc)

            algo_name = method_config["alg_name"]
            try:
                editing_hparams = hparams_mapping[algo_name]
            except KeyError:
                print(f"WARN: {algo_name} has no fitting HParams defined! Skipping Method....")
                continue

            ee_hparams = editing_hparams.from_hparams(path_to_conf)
            self.intervention_controller.register_method(EasyEditInterventionMethod(
                self.intervention_controller,
                ee_hparams
            ))

        self.intervention_controller.register_metric(
            NormOfWeightUpdate(
                self.intervention_controller
            )
        )

        self.intervention_controller.register_metric(
            Perplexity(
                self.intervention_controller
            )
        )

        self.intervention_controller.register_metric(
            DenominatorOfROMEUpdate(
                self.intervention_controller
            )
        )

        self.intervention_controller.register_metric(
            CausalTrace(
                self.intervention_controller
            )
        )

        self.intervention_controller.register_metric(
            Efficacy(
                self.intervention_controller
            )
        )

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

    def get_projections(self, name, layer, dim):
        return self.intervention_controller.get_projections(name, int(layer), int(dim))
