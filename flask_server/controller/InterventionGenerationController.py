import torch
import numpy as np


class InterventionGenerationController:
    def __init__(self, model_wrapper, config):
        """
        Controller, responsible for managing Intervention-Methods, their Interventions, Feature-Scores and Generation
        :type model_wrapper: TransformerModelWrapper
        :param model_wrapper: Model Wrapper, wrapping a Transformer-like LLM
        :param config: Configuration of Backend (jsonnet)
        :type config: dict
        """
        self.model_wrapper = model_wrapper
        self.config = config

        self.interventions = []
        self.intervention_methods = []

        # Pre-Intervention-Metrics are calculated using the LLM without applied Interventions
        self.pre_intervention_metrics = []
        # Post-Intervention-Metrics are calculated using the LLM with applied Interventions
        self.post_intervention_metrics = []

        self.original_weights = {}
        for param_name, param in self.model_wrapper.model.named_parameters():
            # Exclude Embedding
            if "embed" in param_name.lower():
                continue
            self.original_weights[param_name] = param.detach().clone().cpu()

    def register_method(self, method):
        """
        Registers an Intervention Method to this Controller
        :type method: InterventionMethod
        :param method: Intervention Method to register
        """
        self.intervention_methods.append(method)

    def register_metric(self, metric, metric_type):
        """
        Registers a new Metric to this Controller
        :type metric: MetricItem
        :param metric: Metric to register
        :type metric_type: "pre" | "post"
        :param metric_type: Is this Metric a Pre-/Post-Intervention-Metric
        """
        if metric_type == "pre":
            self.pre_intervention_metrics.append(metric)
        elif metric_type == "post":
            self.post_intervention_metrics.append(metric)
        else:
            raise ValueError(f"Parameter metric_type is {metric_type} which is not 'post' or 'pre'")

    def set_interventions(self, interventions):
        """
        Sets the Interventions and distributes to the fitting Intervention Method
        :type interventions: list
        :param interventions: Interventions ot set
        """
        self.clear_interventions()

        self.interventions = interventions

        for intervention in self.interventions:
            intervention_type = intervention["type"]
            intervention_layer = intervention["layer"]

            fitting_method_found = False
            for method in self.intervention_methods:
                if intervention_type == method.get_name() and intervention_layer == method.layer:
                    method.add_intervention(intervention)
                    fitting_method_found = True

            if not fitting_method_found:
                raise AttributeError(f"Intervention <{intervention}> has no fitting Intervention-Method!")

    def clear_interventions(self):
        """
        Clears all Interventions
        """
        self.interventions = []

        for method in self.intervention_methods:
            method.clear_interventions()

    def setup_intervention_hooks(self, prompt):
        """
        Installs the Hooks from the Intervention Methods to the LLM.
        Implementation Logic of Intervention Methods, that use Hooks here.
        :type prompt: str
        :param prompt: Prompt, the Model is run on after setup of Hooks
        """
        for method in self.intervention_methods:
            if len(method.interventions) == 0:
                continue
            method.setup_intervention_hooks(prompt)

    def transform_model(self, prompt):
        """
        Performs the Transformation of the Model's Weights, as defined by the Interventions.
        Implementation Logic of Intervention Methods, that use Model Transformation here.
        :type prompt: str
        :param prompt: Prompt, the Model is run on after Transformation
        """
        # Sort Methods supporting only one Layer from late to early Layers, other Methods are processed after
        # ROMEIntervention won't work else when using multiple Interventions of _different_ ROMEIntervention-Instances
        sorted_methods = sorted(
            self.intervention_methods,
            key=lambda item: item.layer,
            reverse=True
        )
        for method in sorted_methods:
            nonzero_interventions = list(filter(lambda x: x["coeff"] > 0.0, method.interventions))
            if len(nonzero_interventions) == 0:
                continue
            method.transform_model(prompt)

    def restore_original_model(self):
        """
        Restores the original Model's Weights. Inverse of transform_model.
        This Function is inspired from the my-rome/notebooks/rome.ipynb-Notebook of https://github.com/aip-hd-research/my-rome
        """
        if self.original_weights is not None:
            with torch.no_grad():
                for key, original_value in self.original_weights.items():
                    for param_name, param in self.model_wrapper.model.named_parameters():
                        if param_name == key:
                            param[...] = original_value.to(self.model_wrapper.device)

    def get_weight_deltas(self, layers=None):
        deltas = {}

        if self.original_weights is None:
            return {}

        if type(layers) is int:
            layers_list = [layers]
        elif type(layers) is list:
            layers_list = layers
        else:
            raise ValueError("Parameter layers of InterventionGenerationController.get_weight_deltas isn't int or list")

        for layer in layers_list:
            with torch.no_grad():
                for key, original_value in self.original_weights.items():
                    for param_name, param in self.model_wrapper.model.named_parameters():
                        if param_name == key and layer is None:
                            # Deltas of all Layers are calculated
                            deltas[param_name] = original_value - param.detach().clone().cpu()
                        elif param_name == key and f".{layer}." in param_name:
                            # Deltas only of requested Layers are calculated
                            deltas[param_name] = original_value - param.detach().clone().cpu()

        return deltas

    def get_manipulated_layers(self, intervention_method_name=None):
        # If no keyword, find all LLM-Layers that have Interventions attached
        if intervention_method_name is None:
            manip_layers = map(lambda x: x.layer, self.intervention_methods)
        # Else find all LLM-Layers with Interventions where name of InterventionMethod matches intervention_method_name
        else:
            manip_layers = map(
                lambda x: x.layer
                if intervention_method_name == x.get_representation()
                else None,
                self.intervention_methods
            )

        # Filter None's
        manip_layers = filter(lambda x: x is not None, manip_layers)

        # Remove Duplicates
        return list(set(manip_layers))

    def generate(self, prompt, generate_k):
        """
        Generate Text using the Model Wrapper and the set up Interventions.
        Used in the Generate-Feature.
        :rtype: str
        :type prompt: str
        :type generate_k: int
        :param prompt: Prompt, used for Generation
        :param generate_k: Tokens to generate
        :return: Generated Text
        """
        # Call Model-Editing Interventions
        self.transform_model(prompt)
        # Setup Intervention-Hooks
        self.setup_intervention_hooks(prompt)

        response_dict = {}
        tokens = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.model_wrapper.device)
        greedy_output = self.model_wrapper.model.generate(**tokens, max_length=generate_k + len(tokens['input_ids'][0]))
        greedy_output = self.model_wrapper.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        response_dict['generate_text'] = greedy_output

        # Clear Intervention-Hooks and restore original Model (Pre-Transformation)
        self.model_wrapper.clear_hooks()
        self.restore_original_model()

        return response_dict

    def get_token_scores(self, prompt):
        """
        Generates the Token-Scores.
        This Method obtains information on the Next-Token-Prediction of a given Prompt. Used in the Trace-Feature.
        :rtype: dict
        :type prompt: str
        :param prompt: Prompt, used to calculate the Features/Token-Scores
        :return: Token-Scores
        """
        # Assemble API-Response-Dict
        rv_dict = {'prompt': prompt, 'layers': []}

        # Get Frontend-representations of all Interventions
        for method in self.intervention_methods:
            rv_dict['layers'] += method.get_api_layers()

        # Generate Next-Token-Logits for Metric-Parameters (Pre-Intervention)
        tokenizer_output = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens = tokenizer_output["input_ids"].to(self.model_wrapper.device)
        raw_model_output = self.model_wrapper.model(tokens)[0].detach().clone().cpu()
        token_logits = raw_model_output[0]

        # Apply Pre-Intervention-Metrics
        for metric in self.pre_intervention_metrics:
            # Retrieve additional Parameters for this Metric
            additional_params = metric.parameters.return_parameters_object()

            # Calculate Metric and append its Frontend-Layers to API-Response
            rv_dict["layers"] += metric.get_api_layers(prompt, token_logits, additional_params=additional_params)

        # Apply Interventions
        self.transform_model(prompt)
        self.setup_intervention_hooks(prompt)

        # Generate Next-Token-Logits for Metric-Parameters (Post-Intervention)
        raw_model_output = self.model_wrapper.model(tokens)[0].detach().clone().cpu()
        token_logits = raw_model_output[0]

        # Apply Post-Intervention-Metrics
        for metric in self.post_intervention_metrics:
            # Retrieve additional Parameters for this Metric
            additional_params = metric.parameters.return_parameters_object()

            # Calculate Metric and append its Frontend-Layers to API-Response
            rv_dict["layers"] += metric.get_api_layers(prompt, token_logits, additional_params=additional_params)

        # Clear Hooks and restore original Model
        self.model_wrapper.clear_hooks()
        self.restore_original_model()

        # Replace inf's and NaN's with 0
        def replace_inf_nan(iterable):
            if isinstance(iterable, dict):
                for key, value in iterable.items():
                    iterable[key] = replace_inf_nan(value)
            elif isinstance(iterable, list):
                for idx, value in enumerate(iterable):
                    iterable[idx] = replace_inf_nan(value)
            elif isinstance(iterable, float) and (np.isnan(iterable) or np.isinf(iterable)):
                return 0

            return iterable

        return replace_inf_nan(rv_dict)

    def get_projections(self, type, layer, dim):
        """
        Projects Features (their representation as Vectors) to actual Tokens.
        Used in the Value-Vector-Details Feature.
        :rtype: dict
        :type type: str
        :type layer: int
        :type dim: int
        :param type: Type of Intervention Method (Name of the Class, the Intervention Method is implemented in)
        :param layer: Layer Index
        :param dim: Dimension/Index of the Feature ot get Projections from
        :return: Dict of Projections of Features to Tokens
        """
        for method in self.intervention_methods:
            if type != method.get_name() or layer != method.layer:
                continue
            rv = method.get_projections(layer=layer, dim=dim)
            return rv
