from abc import ABC, abstractmethod

class InteractionItem(ABC):
    def __init__(self, controller):
        self.controller = controller
        self.model_wrapper = self.controller.model_wrapper
        self.config = self.controller.config

    def get_name(self):
        """
        Set a custom String for Representation of this Method in Frontend
        :return: Representation
        """
        return self.__class__.__name__

    @abstractmethod
    def get_frontend_items(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_api_layers(self, *args, **kwargs):
        """
        Returns the Frontend-Representation
        :return: Dict of Items to be represented in the Frontend
        """
        pass

    def get_token_scores(self, prompt):
        """
        Generates the Token-Scores.
        This Method obtains information on the Next-Token-Prediction of a given Prompt. Used in the Trace-Feature.
        :type prompt: str
        :param prompt: Prompt, used to calculate the Features/Token-Scores
        :return: Response
        """



class MetricItem(InteractionItem):
    def __init__(self, controller):
        super().__init__(controller)

    @abstractmethod
    def get_text_outputs(self, token_logits):
        pass

    def get_frontend_items(self, token_logits):
        return {
            "text_outputs": self.get_text_outputs(token_logits)
        }

    def get_api_layers(self, token_logits):
        response_dict = [
            {
                "layer": self.__getattribute__("layer") if hasattr(self, "layer") else -1,
                "type": self.get_name()
            }
        ]

        frontend_items = self.get_frontend_items(token_logits)
        for key in frontend_items.keys():
            response_dict[0][key] = frontend_items[key]

        return response_dict


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
