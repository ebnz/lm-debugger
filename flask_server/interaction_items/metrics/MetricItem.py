from abc import ABC, abstractmethod
from ..InteractionItem import InteractionItem


class MetricParameters:
    def __init__(self, metric):
        self.metric = metric

        self.returned_parameters = []

        self.parameters_retrieval_functions = {
            "weight_deltas": lambda: self.metric.controller.get_weight_deltas(
                layers=self.metric.controller.get_manipulated_layers()
            )
        }

    def need_parameter(self, parameter):
        if parameter not in self.parameters_retrieval_functions.keys():
            raise KeyError(f"No Parameter-Retrieval-Function defined for Parameter {parameter}")

        self.returned_parameters.append(parameter)

        return self

    def return_parameters_object(self):
        parameters_object = {}

        for key in self.returned_parameters:
            parameters_object[key] = self.parameters_retrieval_functions[key]()

        return parameters_object


class MetricItem(InteractionItem):
    def __init__(self, controller):
        super().__init__(controller)
        self.parameters = MetricParameters(self)

    @abstractmethod
    def get_text_outputs(self, token_logits, additional_params=None):
        pass

    def get_frontend_items(self, token_logits, additional_params=None):
        return {
            "text_outputs": self.get_text_outputs(token_logits, additional_params=additional_params)
        }

    def get_api_layers(self, token_logits, additional_params=None):
        response_dict = [
            {
                "layer": -1,
                "type": self.get_name()
            }
        ]

        frontend_items = self.get_frontend_items(token_logits, additional_params=additional_params)
        for key in frontend_items.keys():
            response_dict[0][key] = frontend_items[key]

        return response_dict

