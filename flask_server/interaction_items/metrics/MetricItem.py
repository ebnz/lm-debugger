from abc import abstractmethod
from enum import Enum, auto
from ..InteractionItem import InteractionItem
from ...utils import round_struct_recursively


class Attributes(Enum):
    WEIGHT_DELTAS = auto()
    INTERVENTIONS = auto()
    MANIPULATED_LAYERS = auto()


class MetricParameters:
    def __init__(self, metric):
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
        if parameter not in self.parameters_retrieval_functions.keys():
            raise KeyError(f"No Parameter-Retrieval-Function defined for Parameter {parameter}")

        if parameter not in self.returned_parameters:
            self.returned_parameters.append(parameter)

        return self

    def return_parameters_object(self):
        parameters_object = {}

        for key in self.returned_parameters:
            parameters_object[key.name] = self.parameters_retrieval_functions[key]()

        return parameters_object


class MetricItem(InteractionItem):
    def __init__(self, controller):
        super().__init__(controller)
        self.parameters = MetricParameters(self)
        self.config = {}

    @abstractmethod
    def get_text_outputs(self, prompt, token_logits, pre_hook_rv=None, **kwargs):
        pass

    def pre_intervention_hook(self, prompt, **kwargs):
        pass

    def get_frontend_items(self, prompt, token_logits, pre_hook_rv=None, **kwargs):
        return {
            "text_outputs": self.get_text_outputs(prompt, token_logits, pre_hook_rv=pre_hook_rv, **kwargs)
        }

    def get_api_layers(self, prompt, token_logits, pre_hook_rv=None, **kwargs):
        response_dict = [
            {
                "layer": -1,
                "type": self.get_name(),
                "docstring": self.__doc__ if self.__doc__ is not None else "This Metric lacks a Docstring.",
                **self.get_frontend_items(prompt, token_logits, pre_hook_rv=pre_hook_rv, **kwargs)
            }
        ]

        return round_struct_recursively(response_dict)
