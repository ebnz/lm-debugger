from abc import ABC, abstractmethod
from ..InteractionItem import InteractionItem


class InterventionMethod(InteractionItem):
    def __init__(self, controller, layers):
        """
        Represents a generic Intervention Method.
        :type controller: InterventionGenerationController
        :type layers: list
        :param controller: InterventionGenerationController, the Intervention Method is applied to
        :param layers: Layers, supported by this Intervention Method
        """
        super().__init__(controller)

        self.layers = layers
        self.interventions = []

    @abstractmethod
    def get_text_inputs(self):
        pass

    def get_frontend_items(self, layer, prompt, *args, **kwargs):
        return {
            "text_inputs": self.get_text_inputs()
        }

    def get_api_layers(self, prompt):
        response_dict = [
            {
                "layer": layer,
                "type": self.get_name(),
                **self.get_frontend_items(layer, prompt)
            } for layer in self.layers
        ]

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
    def setup_intervention_hook(self, intervention, prompt):
        """
        Installs the Hook, according to the Intervention to the LLM.
        Implementation Logic of Intervention Methods, that use Hooks here.
        :type intervention: dict
        :param intervention: Intervention, to be applied to the Model
        :type prompt: str
        :param prompt: Prompt, the Model is run on after setup of Hooks
        """
        pass

    @abstractmethod
    def transform_model(self, intervention):
        """
        Performs the Transformation of the Model's Weights, as defined by the given Intervention.
        Implementation Logic of Intervention Methods, that use Model Transformation here.
        :param intervention: Intervention to apply
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
