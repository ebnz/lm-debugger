from TokenScoreIntervention import TokenScoreInterventionMethod

class ROMEIntervention(TokenScoreInterventionMethod):
    def __init__(self, model_wrapper, args, rome_hparams):
        super().__init__(model_wrapper, args, rome_hparams.layers, True)

    def setup_intervention_hooks(self, prompt):
        pass


