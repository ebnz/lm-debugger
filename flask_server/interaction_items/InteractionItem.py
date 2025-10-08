from abc import ABC, abstractmethod
from ..controller.InterventionGenerationController import InterventionGenerationController


class InteractionItem(ABC):
    def __init__(self, controller: InterventionGenerationController):
        """
        Abstract Base Class for each Element that shows up in the application (e.g. Intervention Method, Metric)
        :param controller: Controller, the Intervention Method is applied to
        """
        self.controller = controller
        self.model_wrapper = self.controller.model_wrapper
        self.config = self.controller.config

    def get_name(self):
        """
        Set a custom String for Representation of this Method in Frontend
        :return: Representation-String
        """
        return self.__class__.__name__

    """
    Frontend Definitions
    """
    @abstractmethod
    def get_frontend_items(self, *args, **kwargs):
        """
        Returns the Items that show up in a single Layer of the Frontend (e.g. Text In-/Output, Tabular Data, ...).
        One Layer of the Frontend can have multiple Layers in the Frontend
        :return: Dict of all Frontend Items
        """
        pass

    @abstractmethod
    def get_api_layers(self, *args, **kwargs):
        """
        Collects all Layers of one Interaction Item of the Frontend.
        One Interaction Item can have multiple Layers in the Frontend.
        :return: List of all Layers (Dict) to be represented in the Frontend
        """
        pass
