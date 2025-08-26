class InteractionItem:
    def __init__(self, controller, args, layer=0):
        self.controller = controller
        self.model_wrapper = self.controller.model_wrapper
        self.args = args
        self.layer = layer

    def get_representation(self):
        """
        Set a custom String for Representation of this Method in Frontend
        :return: Representation
        """
        return self.__class__.__name__

    def get_frontend_representation(self):
        return {}

    def get_token_scores(self, prompt):
        """
        Generates the Token-Scores.
        This Method obtains information on the Next-Token-Prediction of a given Prompt. Used in the Trace-Feature.
        :type prompt: str
        :param prompt: Prompt, used to calculate the Features/Token-Scores
        :return: Response
        """
        response_dict = {
            "layers": [
                {
                    "layer": self.layer,
                    "type": self.get_representation()
                }
            ]
        }

        for key in self.get_frontend_representation().keys():
            frontend_representation = self.get_frontend_representation()
            response_dict["layers"][0][key] = frontend_representation[key]

        return response_dict


class MetricItem(InteractionItem):
    def __init__(self, controller, args, layer=0):
        super().__init__(controller, args, layer=layer)
        self.metric_value = 0

    def calculate_metric(self, token_logits):
        self.metric_value = 0


class InterventionMethod(InteractionItem):
    def __init__(self, controller, args, layer):
        """
        Represents a generic Intervention Method.
        :type controller: InterventionGenerationController
        :type args: pyhocon.config_tree.ConfigTree
        :type layer: int
        :param controller: InterventionGenerationController, the Intervention Method is applied to
        :param args: Configuration-Options from LM-Debugger++'s JSONNET-Config File
        :param layer: Layer, supported by this Intervention Method
        """
        super().__init__(controller, args, layer=layer)

        self.interventions = []

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

    def get_frontend_representation(self):
        """
        Returns the Frontend-Representation
        :return: Dict of Items to be represented in the Frontend
        """
        return {}

    def setup_intervention_hooks(self, prompt):
        """
        Installs the Hooks from the Interventions to the LLM.
        Implementation Logic of Intervention Methods, that use Hooks here.
        :type prompt: str
        :param prompt: Prompt, the Model is run on after setup of Hooks
        :return: Exit Code
        """
        print(f"WARN: Intervention-Method <{self}> has no implemented <setup_intervention_hooks>")
        return -1

    def transform_model(self, prompt):
        """
        Performs the Transformation of the Model's Weights, as defined by the Interventions.
        Implementation Logic of Intervention Methods, that use Model Transformation here.
        :param prompt: Prompt, the Model is run on after Transformation
        :return:
        """
        print(f"WARN: Intervention-Method <{self}> has no implemented <transform_model>")
        return -1

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
        print(f"WARN: Intervention-Method <{self}> has no implemented <get_projections>")
        return -1
