import torch


class ModelWeightCache:
    def __init__(self, controller):
        self.controller = controller
        self.model_weight_cache = {}

    def cache_key_hash(self, interventions: dict):
        cache_key = str(
            list(
                filter(
                    lambda x: x["type"] != "LMDebuggerIntervention" and x["coeff"] > 0,
                    interventions
                )
            )
        )

        return cache_key

    def check_key(self, key):
        return self.cache_key_hash(key) in self.model_weight_cache.keys()

    def get_value(self, key):
        if not self.check_key(key):
            raise ValueError(f"Key <{key}> not in Cache")

        return self.model_weight_cache[self.cache_key_hash(key)]

    def set_value(self, key, value):
        self.model_weight_cache[key] = value

    def input_weights_into_model(self, model_wrapper, interventions):
        if not self.check_key(interventions):
            raise ValueError(f"Key <{interventions}> not in Cache")

        weights_from_cache = self.get_value(interventions)
        with torch.no_grad():
            for key, original_value in weights_from_cache.items():
                for param_name, param in model_wrapper.model.named_parameters():
                    if param_name == key:
                        param[...] = original_value.to(model_wrapper.device)

    def set_value_from_model(self, model_wrapper, interventions):
        weights_to_cache = {}
        manipulated_layers = self.controller.get_manipulated_layers()
        for param_name, param in model_wrapper.model.named_parameters():
            # Exclude Embedding
            if "embed" in param_name.lower():
                continue
            if any([f".{checked_layer}." in param_name for checked_layer in manipulated_layers]):
                weights_to_cache[param_name] = param.detach().clone().cpu()

        self.set_value(self.cache_key_hash(interventions), weights_to_cache)
