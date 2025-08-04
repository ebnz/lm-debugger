class InterventionMethod:
    def __init__(self, model_wrapper, args, supported_layers):
        """
        Represents a generic Intervention Method.
        :type model_wrapper: transformer_models.TransformerModelWrapper
        :type args: pyhocon.config_tree.ConfigTree
        :type supported_layers: list[int]
        :param model_wrapper: Model Wrapper, the Intervention Method is applied to
        :param args: Configuration-Options from LM-Debugger++'s JSONNET-Config File
        :param supported_layers: Layers, supported by this Intervention Method
        """
        self.args = args
        self.model_wrapper = model_wrapper
        self.TOP_K = self.args.top_k_tokens_for_ui

        # Intervention-specific Variables
        self.supported_layers = supported_layers

        self.interventions = []

    def get_representation(self):
        """
        Set a custom String for Representation of this Method in Frontend
        :return: Representation
        """
        return self.__class__.__name__

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

    def get_token_scores(self, prompt):
        """
        Generates the Token-Scores.
        This Method obtains information on the Next-Token-Prediction of a given Prompt. Used in the Trace-Feature.
        :type prompt: str
        :param prompt: Prompt, used to calculate the Features/Token-Scores
        :return: Exit Code
        """
        print(f"WARN: Intervention-Method <{self}> has no implemented <get_token_scores>")
        return -1

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
