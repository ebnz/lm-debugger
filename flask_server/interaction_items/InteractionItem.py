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