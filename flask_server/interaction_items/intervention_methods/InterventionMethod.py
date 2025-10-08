from abc import ABC, abstractmethod
from ..InteractionItem import InteractionItem
from ...controller.InterventionGenerationController import InterventionGenerationController


class InterventionMethod(InteractionItem):
    def __init__(self, controller: InterventionGenerationController, layers: list[int]):
        """
        Abstract Base Class that represents a generic Intervention Method.
        :param controller: InterventionGenerationController, the Intervention Method is applied to
        :param layers: Layers, supported by this Intervention Method
        """
        super().__init__(controller)

        self.layers = layers
        self.interventions = []

    """
    Frontend Definitions
    """
    @abstractmethod
    def get_text_inputs(self):
        """
        Returns the text inputs, an Intervention Method can have
        """
        pass

    def get_frontend_items(self, layer: int, prompt: str, *args, **kwargs):
        return {
            "text_inputs": self.get_text_inputs()
        }

    def get_api_layers(self, prompt: str):
        response_dict = [{
            "layer": layer,
            "type": self.get_name(),
            "docstring": self.__doc__ if self.__doc__ is not None else "This Intervention Method lacks a Docstring.",
            **self.get_frontend_items(layer, prompt)
        } for layer in self.layers]

        return response_dict

    """
    Intervention Add/Set/Clear
    """
    def add_intervention(self, intervention: dict):
        """
        Add an Intervention to this Intervention Method
        :param intervention: Intervention to add
        """
        self.interventions.append(intervention)

    def set_interventions(self, interventions: list[dict]):
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

    """
    Attach Interventions / Transform Model
    """
    @abstractmethod
    def setup_intervention_hook(self, intervention: dict, prompt: str):
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
    def transform_model(self, intervention: dict):
        """
        Performs the Transformation of the Model's Weights, as defined by the given Intervention.
        Implementation Logic of Intervention Methods, that use Model Transformation here.
        :param intervention: Intervention to apply
        """
        pass

    """
    Intervention Feature Details
    """
    @abstractmethod
    def get_projections(self, dim: int, *args, **kwargs):
        """
        Projects Features (their representation as Vectors) to actual Tokens.
        Used in the Value-Vector-Details Feature.
        :param dim: Dimension/Index of the Feature ot get Projections from
        :param args
        :param **kwargs
        :return: Exit Code
        """
        pass
