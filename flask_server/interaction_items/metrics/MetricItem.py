from abc import ABC, abstractmethod
from ..InteractionItem import InteractionItem

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
                "layer": -1,
                "type": self.get_name()
            }
        ]

        frontend_items = self.get_frontend_items(token_logits)
        for key in frontend_items.keys():
            response_dict[0][key] = frontend_items[key]

        return response_dict

