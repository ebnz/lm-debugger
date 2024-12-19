class TokenScoreInterventionMethod:
    def __init__(self, model_wrapper, args, supported_layers):
        self.args = args
        self.model_wrapper = model_wrapper
        self.TOP_K = self.args.top_k_tokens_for_ui

        # Intervention-specific Variables
        self.supported_layers = supported_layers

        self.interventions = []

    def add_intervention(self, intervention):
        self.interventions.append(intervention)

    def set_interventions(self, interventions):
        self.interventions = interventions

    def clear_interventions(self):
        self.interventions = []

    def get_token_scores(self, prompt):
        raise NotImplementedError(f"Intervention-Method <{self}> has no implemented <get_token_scores>")

    def setup_intervention_hooks(self, prompt):
        raise NotImplementedError(f"Intervention-Method <{self}> has no implemented <setup_intervention_hooks>")

    def transform_model(self):
        raise NotImplementedError(f"Intervention-Method <{self}> has no implemented <transform_model>")

    def restore_original_model(self):
        raise NotImplementedError(f"Intervention-Method <{self}> has no implemented <restore_original_model>")

    def get_projections(self, dim, *args, **kwargs):
        raise NotImplementedError(f"Intervention-Method <{self}> has no implemented <get_projections>")
