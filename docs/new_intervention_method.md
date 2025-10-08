# Implementing a new Intervention Method

Implementing a new Intervention Method can be done by creating a subclass of the Abstract Base Class `InterventionMethod`.

### Types of Intervention Methods
An Intervention Method can either be implemented as a **Hook-based Intervention Method** or a **Model-Transform Intervention Method**. 

#### General
Each Method should set the Attribute `self.layers = [0, 1, 2]` to the layer indices, the Method mutates. This allows to index Intervention Method in the Frontend and also cache a Model's Weights to later undo Model-Transformations. 

#### Hook-based Intervention Method

A **Hook-based Intervention Method** uses Features of a Model. Interventions place Hooks on specific Features in the computational Graph of the Transformer Model to directly mutate the activation of the Feature. 

A **Hook-based Intervention Method** implements the following Methods:
* `setup_intervention_hook`
  * Installs a Hook, given as a Parameter
  * Use Method `self.model_wrapper.setup_hook` to install Hooks
* `get_projections`
  * Returns a Feature's projected Tokens and Logit Values

#### Model-Transform Intervention Method

A **Model-Transform Intervention Method** transforms the Model-Weights of the Transformer Model. One Intervention could execute an arbitrary Model-Editing-Algorithm once. 

A **Model-Transform Intervention Method** implements the following Methods:
* `get_text_inputs`
  * Returns a dict of Names of Text-Inputs (Keys) and standard-inputs (Values)
* `transform_model`
  * Performs the Transformation of the Model's Weights based on a given Intervention

### Detailed Methods Explanation

#### get_name
By default, we use the name of this Subclass as the Name of this Intervention Method. 
By overriding this Method, a custom name can be set. 

Examples: 
* `EasyEditInterventionMethod`

#### get_text_inputs
Returns a dict of Text Inputs, which are used to define an Intervention. The defined Text Inputs show up in the UI. 

The following dict defines three Text-Inputs, that have empty standard-values. 
```python
def get_text_inputs(self):
    return {
        "prompt": "",
        "subject": "",
        "target": ""
    }
```

A set Intervention will have the same structure with filled out Dict-Values. Exemplary Intervention: 

```python
{
    "layer": 5, 
    "type": "ExampleInterventionMethod", 
    "text_inputs": {
        "prompt": "{} is a",
        "subject": "Barack Obama",
        "target": "human"
    },
    "coeff": 1
}
```

#### transform_model
Transforms the Model according to a given Intervention. Exemplary implementation: 

```python
def transform_model(self, intervention):
    # Skip disabled Interventions
    if intervention["coeff"] <= 0.0:
        return

    request = [{
        "prompt": intervention["text_inputs"]["prompt"],
        "subject": intervention["text_inputs"]["prompt"],
        "target_new": intervention["text_inputs"]["target"]
    }]

    rv = self.invoke_method(
        self.model_wrapper.model,
        self.model_wrapper.tokenizer,
        request,
        self.ee_hparams,
        copy=False
    )

    if isinstance(rv, tuple):
        edited_model = rv[0]
    else:
        edited_model = rv

    self.model_wrapper.model = edited_model
```

#### setup_intervention_hook
Sets an Intervention Hook according to a given Intervention. Exemplary implementation: 

```python
def setup_intervention_hooks(self, intervention: dict, prompt: str):
    def hook_mlp_acts(module, input, output):
        activation_vector = output
        f = autoencoder.forward_encoder(activation_vector)
        f[::, ::, 1234] = 42
        x_hat = autoencoder.forward_decoder(f)
        return x_hat

    self.model_wrapper.setup_hook(
        hook_mlp_acts,
        "model.layers.3.mlp"
    )
```

Set Hooks are automatically cleared after usage. 

#### get_projections
The results of `get_projections` are shown in the side menu (ValueDetailsPanel), once a Feature of an Intervention is clicked. 

```python
def get_projections(self, dim, *args, **kwargs):
    return {
        "dim": 1278,
        "layer": 42,
        "top_k": [{
            "logit": 1.23,
            "token": "_my"
        }, ...]
    }
```
