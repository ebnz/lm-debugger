# Implementing a new Metric

Implementing a new Metric can be done by creating a subclass of the Abstract Base Class `MetricItem`.

Examples: 
* `CausalTrace`
* `Perplexity`
* `NormOfWeightUpdate`
* `DenominatorOfROMEUpdate`

### What do I need to implement?
First of all, a basic Metric must only implement the Method `get_text_outputs` which calculates the Metric values and returns the values. 

Additionally, it may be required to implement the Method `pre_intervention_hook`, which is a Hook, that's called before the Interventions are applied to the Transformer. May be required for some Metrics, such as ROME-Efficacy. 

### Detailed Methods Explanation
#### get_text_outputs
This Method returns the output of the Metric. 

The following Listing shows the implementation of `NormOfWeightUpdate`: 
```python
def get_text_outputs(self, prompt, token_logits, pre_hook_rv=None, MANIPULATED_LAYERS=None, WEIGHT_DELTAS=None):
    metric_values = {}

    for layer in MANIPULATED_LAYERS:
        if layer not in range(self.controller.config.num_layers):
            continue

        down_descriptor = self.controller.config.layer_mappings["mlp_down_proj"].format(layer) + ".weight"

        metric_values[f"Layer {layer}"] = torch.linalg.matrix_norm(WEIGHT_DELTAS[down_descriptor]).item()

    return metric_values
```

Parameters: 
* `prompt`: The Prompt, used in this Trace-Call
* `token_logits`: Output-Distribution of the Transformer over all inputted Tokens (autoregressive run). Inputting $16$ Tokens and having Vocabulary Size of $50000$, this is a Matrix of dimension $16 \times 50000$
* `pre_hook_rv`: Return Value of the Method `pre_intervention_hook`
* `**kwargs`: Additionally requested Keyword-Arguments (ref. Section `Additional Keyword Arguments`)

#### pre_intervention_hook
This Method implements routines, that are executed before any Interventions are applied to the Transformer Model. 

**The Return Value of this Method is a Parameter of `get_text_outputs`**

Theoretically, it's possible to calculate the entirety of a Metric in here (see Metric `CausalTrace`). 

### Additional Keyword-Arguments
Additional Keyword-Arguments, needed in a Metric's calculation can be requested by the following call in a Metric's constructor: 

```python
class NormOfWeightUpdate(MetricItem):
    """
    Calculates the 2-Norm of the Weight-Delta-Matrix of each Knowledge-Editing Intervention.
    High values imply updates with large magnitude and may corrupt the LLM.
    Hyperparameters of Intervention Methods influence the magnitude of the update and thus it's 2-Norm.
    """
    def __init__(self, controller):
        super().__init__(controller)

        # Request for additional Parameters
        self.parameters.need_parameter(Attributes.WEIGHT_DELTAS)
        self.parameters.need_parameter(Attributes.MANIPULATED_LAYERS)
```

By default, the following Attributes are available: 
* `WEIGHT_DELTAS`
* `INTERVENTIONS`
* `MANIPULATED_LAYERS`

If you want to define your own additional Keyword-Arguments, you can do so by adding the attribute's name to the Enum `Attributes` and defining a retrieval function in the class `MetricParameters` in `MetricItem.py`. 

```python
class Attributes(Enum):
    WEIGHT_DELTAS = auto()
    INTERVENTIONS = auto()
    MANIPULATED_LAYERS = auto()


class MetricParameters:
    def __init__(self, metric):
        """
        Collector of all additionally needed Parameters of a Metric.

        To define a new Parameter:
        * Add descriptor to Attributes-Enum
        * Add getter-Function to self.parameters_retrieval_functions

        To use a Parameter in a Metric:
        * Call self.parameters.need_parameter(ATTRIBUTE)
        * Obtain Parameter via Keyword-Parameter
        """
        self.metric = metric

        self.returned_parameters = []

        self.parameters_retrieval_functions = {
            Attributes.WEIGHT_DELTAS: lambda: self.metric.controller.get_weight_deltas(
                layers=self.metric.controller.get_manipulated_layers()
            ),
            Attributes.INTERVENTIONS: lambda: self.metric.controller.interventions,
            Attributes.MANIPULATED_LAYERS: self.metric.controller.get_manipulated_layers
        }
```