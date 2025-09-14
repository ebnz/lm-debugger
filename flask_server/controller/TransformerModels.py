import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy

class TransformerModelWrapper:
    def __init__(self, model_name, dtype=torch.float16, device="cpu"):
        """
        A Wrapper-Class for a Transformer-like LLM
        :type model_name: transformers.Model
        :type device: str
        :param model_name: str
        :param device: Device for the Model
        """
        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.device = device
        self.to(device)

        self.model_hook_handles = []

    def to(self, device):
        """
        Offloads the Transformer Model to the specified Device.
        :type device: str
        :param device: Device to offload to
        """
        self.device = device
        self.model.to(self.device)

    def generate(self, prompt, top_p=0.9, temperature=1.0, max_new_tokens=500, add_special_tokens=True):
        """
        Generates an answer to a specific prompt with a chosen LLM-Model and Tokenizer.
        :type prompt: str
        :type top_p: float
        :type temperature: float
        :type max_new_tokens: int
        :type add_special_tokens: bool
        :param prompt: The Prompt for the LLM
        :param top_p: top_p Parameter for Beam Searching the LLM
        :param temperature: Randomness Factor
        :param max_new_tokens: Maximum Amount of new Tokens that are generated
        :param add_special_tokens: Whether to add special tokens such as <s>, </s>, [INST], [/INST], ...
        :return: Decoded output of LLM to the given prompt
        """

        # Tokenize input string and send to device on which the model is located (e.g. cuda)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
        input_ids_cuda = inputs["input_ids"].to(self.device)
        attention_mask_cuda = inputs["attention_mask"].to(self.device)

        # Generate output with LLM model
        output = self.model.generate(
            input_ids_cuda,
            attention_mask=attention_mask_cuda,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Return decoded tokens
        return self.tokenizer.decode(output[0])

    def setup_hook(self, hook, module_name, permanent=False):
        """
        Place a Hook in a Transformer-Model.
        :type hook: function
        :type module_name: str
        :type permanent: bool
        :param hook: Hook to install
        :param module_name: Name of the Module, the Hook is placed on
        :param permanent: Whether a call of clear_hooks should remove the Hook
        """
        modules_dict = dict(self.model.named_modules())

        # Retrieve Module from Name and register Hook
        try:
            module = modules_dict[module_name]
        except KeyError:
            raise ValueError(f"Module: <{module_name}> does not exist in Model <{self.model}> "
                             f"and is not registered in layer_aliases of TransformerModelWrapper")

        handle = module.register_forward_hook(hook)

        # Permanent Hooks can't be removed
        if not permanent:
            self.model_hook_handles.append(handle)

    def clear_hooks(self):
        """
        Clear all non-permanent Hooks placed in this Model.
        """
        for hook in self.model_hook_handles:
            hook.remove()
        self.model_hook_handles = []

    # Delegate missing Methods/Attributes to self.model
    def __getattr__(self, name):
        return getattr(self.model, name)
