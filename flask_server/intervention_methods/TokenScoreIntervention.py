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
        print(f"WARN: Intervention-Method <{self}> has no implemented <get_token_scores>")
        return -1

    def setup_intervention_hooks(self, prompt):
        print(f"WARN: Intervention-Method <{self}> has no implemented <setup_intervention_hooks>")
        return -1

    def transform_model(self):
        print(f"WARN: Intervention-Method <{self}> has no implemented <transform_model>")
        return -1

    def get_projections(self, dim, *args, **kwargs):
        print(f"WARN: Intervention-Method <{self}> has no implemented <get_projections>")
        return -1
