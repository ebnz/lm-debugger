import torch

class InterventionGenerationController:
    def __init__(self, model_wrapper, top_k):
        self.model_wrapper = model_wrapper
        self.TOP_K = top_k
        self.interventions = []
        self.intervention_methods = []

        self.original_weights = {}
        for param_name, param in self.model_wrapper.model.named_parameters():
            # Exclude Embedding
            if "embed" in param_name.lower():
                continue
            self.original_weights[param_name] = param.detach().clone().cpu()

    def register_method(self, method):
        self.intervention_methods.append(method)

    def set_interventions(self, interventions):
        self.clear_interventions()

        self.interventions = interventions

        for intervention in self.interventions:
            intervention_type = intervention["type"]
            intervention_layer = intervention["layer"]
            print(f"My Layer: {intervention_layer}")
            fitting_method_found = False
            for method in self.intervention_methods:
                if method.__class__.__name__ == intervention_type and intervention_layer in method.supported_layers:
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

    def transform_model(self, prompt):
        # Sort Methods supporting only one Layer from late to early Layers, other Methods are processed after
        # ROMEIntervention won't work else when using multiple Interventions of _different_ ROMEIntervention-Instances
        sorted_methods = sorted(
            self.intervention_methods,
            key=lambda item: item.supported_layers[0] if len(item.supported_layers) == 1 else 0,
            reverse=True
        )
        for method in sorted_methods:
            method.transform_model(prompt)

    """
    This Function is inspired from the my-rome/notebooks/rome.ipynb-Notebook of https://github.com/aip-hd-research/my-rome
    """
    def restore_original_model(self):
        if self.original_weights is not None:
            with torch.no_grad():
                for key, original_value in self.original_weights.items():
                    for param_name, param in self.model_wrapper.model.named_parameters():
                        if param_name == key:
                            param[...] = original_value.to(self.model_wrapper.device)

    def generate(self, prompt, generate_k):
        # Call Model-Editing Interventions
        self.transform_model(prompt)
        # Setup Intervention-Hooks
        self.setup_intervention_hooks(prompt)

        response_dict = {}
        tokens = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.model_wrapper.device)
        greedy_output = self.model_wrapper.model.generate(**tokens,
                                            max_length=generate_k + len(tokens['input_ids'][0]))
        greedy_output = self.model_wrapper.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        response_dict['generate_text'] = greedy_output

        # Clear Intervention-Hooks and restore original Model (Pre-Transformation)
        self.model_wrapper.clear_hooks()
        self.restore_original_model()

        return response_dict

    def get_token_scores(self, prompt):
        # Apply Interventions
        self.transform_model(prompt)
        self.setup_intervention_hooks(prompt)

        rv_dict = {'prompt': prompt, 'layers': []}
        # Assemble Response-Dict
        for method in self.intervention_methods:
            # Generate Token-Scores
            rv = method.get_token_scores(prompt)
            rv_dict['layers'] += rv['layers']

        # Clear Hooks and restore original Model
        self.model_wrapper.clear_hooks()
        self.restore_original_model()

        # Replace inf's with 0
        def replace_infs(dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    dictionary[key] = replace_infs(value)
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        dictionary[key][idx] = replace_infs(item)
                elif value == float("inf"):
                    dictionary[key] = 0
            return dictionary

        return rv_dict

    def get_projections(self, type, layer, dim):
        for method in self.intervention_methods:
            if type != method.__class__.__name__:
                continue
            if layer not in method.supported_layers:
                continue
            rv = method.get_projections(layer=layer, dim=dim)
            return rv
