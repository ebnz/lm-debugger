from abc import abstractmethod
from enum import Enum, auto

import torch

from ..InteractionItem import InteractionItem
from ...utils import round_struct_recursively


class Attributes(Enum):
    WEIGHT_DELTAS = auto()
    INTERVENTIONS = auto()
    MANIPULATED_LAYERS = auto()


class MetricParameters:
    def __init__(self, metric):
        """
        Collector of all additionally needed Parameters of a Metric.

        To define a new Parameter:
        * Add descriptor to Attributes-Enum
        * Add getter-Function to self.parameters_retrieval_functions

        To use a Parameter in a Metric:
        * Call self.parameters.need_parameter(ATTRIBUTE)
        * Obtain Parameter via Keyword-Parameter
        """
        self.metric = metric

        self.returned_parameters = []

        self.parameters_retrieval_functions = {
            Attributes.WEIGHT_DELTAS: lambda: self.metric.controller.get_weight_deltas(
                layers=self.metric.controller.get_manipulated_layers()
            ),
            Attributes.INTERVENTIONS: lambda: self.metric.controller.interventions,
            Attributes.MANIPULATED_LAYERS: self.metric.controller.get_manipulated_layers
        }

    def need_parameter(self, parameter):
        """
        Request an additional Parameter from inside a MetricItem
        """
        if parameter not in self.parameters_retrieval_functions.keys():
            raise KeyError(f"No Parameter-Retrieval-Function defined for Parameter {parameter}")

        if parameter not in self.returned_parameters:
            self.returned_parameters.append(parameter)

        return self

    def return_parameters_object(self):
        """
        Returns the assembled Parameters-Dict of all Parameters, requested via MetricParameters.need_parameter(ATTRIB)
        """
        parameters_object = {}

        for key in self.returned_parameters:
            parameters_object[key.name] = self.parameters_retrieval_functions[key]()

        return parameters_object


class MetricItem(InteractionItem):
    def __init__(self, controller):
        super().__init__(controller)
        self.parameters = MetricParameters(self)
        self.config = {}

    """
    Frontend Handling
    """
    def get_type(self):
        return "metric"

    @abstractmethod
    def get_text_outputs(self, prompt: str, token_logits: torch.Tensor, pre_hook_rv=None, **kwargs):
        pass

    def get_frontend_items(self, prompt: str, token_logits: torch.Tensor, pre_hook_rv=None, **kwargs):
        return {
            "text_outputs": self.get_text_outputs(prompt, token_logits, pre_hook_rv=pre_hook_rv, **kwargs)
        }

    def get_api_layers(self, prompt: str, token_logits: torch.Tensor, pre_hook_rv=None, **kwargs):
        response_dict = [
            {
                "layer": 0,
                "name": self.get_name(),
                "type": self.get_type(),
                "changeable_layer": self.get_changeable_layer(),
                "docstring": self.__doc__ if self.__doc__ is not None else "This Metric lacks a Docstring.",
                **self.get_frontend_items(prompt, token_logits, pre_hook_rv=pre_hook_rv, **kwargs)
            }
        ]

        return round_struct_recursively(response_dict)

    def pre_intervention_hook(self, prompt: str, **kwargs):
        """
        Method, which is called before any Intervention are applied.
        This way, the Model, ..., can be used to e.g. compare the Metric to a Baseline.
        The Return Value of this Method is given to Frontend-Handlers (e.g. get_text_outputs) as Parameter pre_hook_rv
        """
        pass
