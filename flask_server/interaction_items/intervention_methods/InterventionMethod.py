from abc import ABC, abstractmethod
from ..InteractionItem import InteractionItem

class InterventionMethod(InteractionItem):
    def __init__(self, controller, layer):
        """
        Represents a generic Intervention Method.
        :type controller: InterventionGenerationController
        :type layer: int
        :param controller: InterventionGenerationController, the Intervention Method is applied to
        :param layer: Layer, supported by this Intervention Method
        """
        super().__init__(controller)

        self.layer = layer
        self.interventions = []

    @abstractmethod
    def get_text_inputs(self):
        pass

    def get_frontend_items(self, *args, **kwargs):
        return {
            "text_inputs": self.get_text_inputs()
        }

    def get_api_layers(self):
        response_dict = [
            {
                "layer": self.layer,
                "type": self.get_name()
            }
        ]

        frontend_items = self.get_frontend_items()
        for key in frontend_items.keys():
            response_dict[0][key] = frontend_items[key]

        return response_dict

    def add_intervention(self, intervention):
        """
        Add an Intervention to this Intervention Method
        :param intervention: Intervention to add
        """
        self.interventions.append(intervention)

    def set_interventions(self, interventions):
        """
        Set multiple Interventions at once.
        :param interventions: List of Interventions to set
        """
        self.interventions = interventions

    def clear_interventions(self):
        """
        Clear all Interventions from this Intervention Method.
        """
        self.interventions = []

    @abstractmethod
    def setup_intervention_hooks(self, prompt):
        """
        Installs the Hooks from the Interventions to the LLM.
        Implementation Logic of Intervention Methods, that use Hooks here.
        :type prompt: str
        :param prompt: Prompt, the Model is run on after setup of Hooks
        """
        pass

    @abstractmethod
    def transform_model(self, prompt):
        """
        Performs the Transformation of the Model's Weights, as defined by the Interventions.
        Implementation Logic of Intervention Methods, that use Model Transformation here.
        :param prompt: Prompt, the Model is run on after Transformation
        """
        pass

    @abstractmethod
    def get_projections(self, dim, *args, **kwargs):
        """
        Projects Features (their representation as Vectors) to actual Tokens.
        Used in the Value-Vector-Details Feature.
        :param type: Type of Intervention Method (Name of the Class, the Intervention Method is implemented in)
        :param layer: Layer Index
        :param dim: Dimension/Index of the Feature ot get Projections from
        :param args
        :param **kwargs
        :return: Exit Code
        """
        pass
