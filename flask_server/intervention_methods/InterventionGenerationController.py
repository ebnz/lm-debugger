class InterventionGenerationController:
    def __init__(self, model_wrapper, top_k):
        self.model_wrapper = model_wrapper
        self.TOP_K = top_k
        self.interventions = []
        self.intervention_methods = []

    def register_method(self, method):
        self.intervention_methods.append(method)

    def set_interventions(self, interventions):
        self.clear_interventions()

        self.interventions = interventions

        for intervention in self.interventions:
            intervention_type = intervention["type"]
            fitting_method_found = False
            for method in self.intervention_methods:
                if method.__class__.__name__ == intervention_type:
                    if intervention_type == "SAEIntervention" and intervention["layer"] != method.config["LAYER_INDEX"]:
                        continue
                    method.add_intervention(intervention)
                    fitting_method_found = True
            if not fitting_method_found:
                raise AttributeError(f"Intervention <{intervention}> has no fitting Intervention-Method!")


    def clear_interventions(self):
        self.interventions = []

        for method in self.intervention_methods:
            method.clear_interventions()

    def setup_intervention_hooks(self, prompt):
        for method in self.intervention_methods:
            method.setup_intervention_hooks(prompt)

    def generate(self, prompt, generate_k):
        # Setup Intervention-Hooks
        self.setup_intervention_hooks(prompt)

        response_dict = {}
        tokens = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.model_wrapper.device)
        greedy_output = self.model_wrapper.model.generate(**tokens,
                                            max_length=generate_k + len(tokens['input_ids'][0]))
        greedy_output = self.model_wrapper.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        response_dict['generate_text'] = greedy_output

        self.model_wrapper.clear_hooks()

        return response_dict

    def get_projections(self, type, layer, dim):
        for method in self.intervention_methods:
            if type != method.__class__.__name__:
                continue
            if layer not in method.supported_layers:
                continue
            rv = method.get_projections(layer=layer, dim=dim)
            return rv
