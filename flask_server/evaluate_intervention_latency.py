import torch
import json
import _jsonnet
import pyhocon
import argparse
import time

from tqdm import tqdm

import plotly.express as px

from sparse_autoencoders import CodeLlamaModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from intervention_methods.InterventionGenerationController import InterventionGenerationController
from intervention_methods.LMDebuggerIntervention import LMDebuggerIntervention
from intervention_methods.ROMEIntervention import ROMEIntervention
from intervention_methods.SAEIntervention import SAEIntervention

class InterventionGenerationControllerForTiming(InterventionGenerationController):
    def __init__(self, model_wrapper, top_k):
        super().__init__(model_wrapper, top_k)

    def generate(self, prompt, generate_k):
        # Call Model-Editing Interventions
        self.transform_model(prompt)
        # Setup Intervention-Hooks
        self.setup_intervention_hooks(prompt)

        response_dict = {}
        instructed_prompt = f"[INST]<<SYS>>Explain the following Code Snippet in detail. <</SYS>>{prompt}[/INST]"
        tokens = self.model_wrapper.tokenizer(instructed_prompt, return_tensors="pt")
        tokens.to(self.model_wrapper.device)

        torch.cuda.synchronize()
        start_time = time.time()
        greedy_output = self.model_wrapper.model.generate(**tokens,
                                                          max_length=generate_k + len(tokens['input_ids'][0]))
        torch.cuda.synchronize()
        stop_time = time.time()
        greedy_output = self.model_wrapper.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        response_dict['generate_text'] = greedy_output

        # Clear Intervention-Hooks and restore original Model (Pre-Transformation)
        self.model_wrapper.clear_hooks()
        self.restore_original_model()

        elapsed_time = stop_time - start_time

        return response_dict, elapsed_time


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, help="specify the config file"
)
commandline_args = parser.parse_args()
args = pyhocon.ConfigFactory.from_dict(json.loads(_jsonnet.evaluate_file(commandline_args.config_path)))

model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model_wrapper = CodeLlamaModel(model, tokenizer=tokenizer, device=args.device)

intervention_controller = InterventionGenerationControllerForTiming(model_wrapper, args.top_k_tokens_for_ui)

# Load LMDebuggerIntervention
intervention_controller.register_method(LMDebuggerIntervention(
    model_wrapper,
    args
))

# Load SAEs
for config_path in tqdm(args.sae_paths, desc="Loading SAE-Instances"):
    intervention_controller.register_method(SAEIntervention(
        model_wrapper,
        args,
        config_path
    ))

# Load ROME Instances
for config_path in tqdm(args.rome_paths, desc="Loading ROME-Instances"):
    intervention_controller.register_method(ROMEIntervention(
        model_wrapper,
        args,
        config_path
    ))

labels = [200, 400, 600, 800, 1000]
no_intervention_times = []

ds = load_dataset("bigcode/the-stack-dedup", streaming=True, split="train")
iterator = iter(ds)

for gen_tokens in labels:
    sample = next(iterator)["content"]
    tokens = tokenizer(sample, return_tensors="pt", truncation=True, max_length=64, add_special_tokens=False)
    while len(tokens["input_ids"][0]) < 64:
        sample = next(iterator)["content"]
        tokens = tokenizer(sample, return_tensors="pt", truncation=True, max_length=64, add_special_tokens=False)
    response, elapsed_time = intervention_controller.generate(sample, gen_tokens)

    print(response)
    print(elapsed_time)
    break