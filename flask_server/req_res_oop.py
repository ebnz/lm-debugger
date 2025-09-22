import os
import warnings

import yaml
from tqdm import tqdm

# Controller
from controller.InterventionGenerationController import InterventionGenerationController
from controller.TransformerModels import TransformerModelWrapper

# Metrics
from interaction_items.metrics.ExcessiveWeightDeltasMetric import ExcessiveWeightDeltasMetric
from interaction_items.metrics.PerplexityMetric import PerplexityMetric
from interaction_items.metrics.OutOfDistributionKeysMetric import OutOfDistributionKeysMetric
from interaction_items.metrics.LocalizationVEditingMetric import LocalizationVEditingMetric
from interaction_items.metrics.EfficacyMetric import EfficacyMetric

# Intervention Methods
from interaction_items.intervention_methods.EasyEditInterventionMethod import EasyEditInterventionMethod
from interaction_items.intervention_methods.EasyEdit.easyeditor import (
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    R_ROMEHyperParams,
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    GraceHyperParams,
    WISEHyperParams,
    AlphaEditHyperParams,
)

warnings.filterwarnings('ignore')


class ModelingRequests:
    def __init__(self, config):
        self.config = config

        self.model_wrapper = TransformerModelWrapper(config.model_name, device=config.device)

        self.intervention_controller = InterventionGenerationController(self.model_wrapper, self.config)

        for file_name in tqdm(os.listdir(config.easy_edit_hparams_path), desc="Loading EasyEdit Methods"):
            path_to_conf = os.path.join(config.easy_edit_hparams_path, file_name)
            with open(path_to_conf) as file_desc:
                method_config = yaml.safe_load(file_desc)

            # ToDo: Make dict and add rest
            if method_config["alg_name"] == 'FT':
                editing_hparams = FTHyperParams
            elif method_config["alg_name"] == 'IKE':
                editing_hparams = IKEHyperParams
            elif method_config["alg_name"] == 'KN':
                editing_hparams = KNHyperParams
            elif method_config["alg_name"] == 'MEMIT':
                editing_hparams = MEMITHyperParams
            elif method_config["alg_name"] == 'ROME':
                editing_hparams = ROMEHyperParams
            elif method_config["alg_name"] == "R-ROME":
                editing_hparams = R_ROMEHyperParams
            elif method_config["alg_name"] == 'LoRA':
                editing_hparams = LoRAHyperParams
            elif method_config["alg_name"] == 'MEND':
                editing_hparams = MENDHyperParams
            elif method_config["alg_name"] == 'GRACE':
                editing_hparams = GraceHyperParams
            elif method_config["alg_name"] == 'WISE':
                editing_hparams = WISEHyperParams
            elif method_config["alg_name"] == 'AlphaEdit':
                editing_hparams = AlphaEditHyperParams
            else:
                raise NotImplementedError

            ee_hparams = editing_hparams.from_hparams(path_to_conf)
            self.intervention_controller.register_method(EasyEditInterventionMethod(
                self.intervention_controller,
                ee_hparams
            ))

        self.intervention_controller.register_metric(
            ExcessiveWeightDeltasMetric(
                self.intervention_controller
            )
        )

        self.intervention_controller.register_metric(
            PerplexityMetric(
                self.intervention_controller
            )
        )

        self.intervention_controller.register_metric(
            OutOfDistributionKeysMetric(
                self.intervention_controller
            )
        )

        self.intervention_controller.register_metric(
            LocalizationVEditingMetric(
                self.intervention_controller
            )
        )

        self.intervention_controller.register_metric(
            EfficacyMetric(
                self.intervention_controller
            )
        )

    def request2response(self, req_json_dict):
        prompt = req_json_dict['prompt']
        interventions = req_json_dict['interventions']

        # ToDo: Only one invocation
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
