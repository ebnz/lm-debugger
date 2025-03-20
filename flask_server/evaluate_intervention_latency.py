import os
import gc
import torch
import json
import _jsonnet
import pyhocon
import argparse
import time
import random

from tqdm import tqdm

import plotly.graph_objs as go

from sparse_autoencoders import CodeLlamaModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from intervention_methods.InterventionGenerationController import InterventionGenerationController
from intervention_methods.LMDebuggerIntervention import LMDebuggerIntervention
from intervention_methods.ROMEIntervention import ROMEIntervention
from intervention_methods.SAEIntervention import SAEIntervention

class LMDebuggerInterventionForTiming(LMDebuggerIntervention):
    def __init__(self, model_wrapper, args):
        super().__init__(model_wrapper, args)

    # New Implementation, removes calculation of coeff-Value
    def setup_intervention_hooks(self, prompt):
        if len(self.interventions) <= 0:
            return
        for intervention in self.interventions:
            if intervention['coeff'] > 0:
                new_max_val = intervention["coeff"]
            else:
                new_max_val = 0
            self.set_control_hooks_gpt2({intervention['layer']: [intervention['dim']], },
                                        coef_value=new_max_val)

class InterventionGenerationControllerForTiming(InterventionGenerationController):
    def __init__(self, model_wrapper):
        super().__init__(model_wrapper)

    def generate(self, prompt, generate_k):
        torch.cuda.synchronize()
        intervention_start_time = time.time()

        # Call Model-Editing Interventions
        self.transform_model(prompt)
        # Setup Intervention-Hooks
        self.setup_intervention_hooks(prompt)

        torch.cuda.synchronize()
        intervention_stop_time = time.time()

        # Clean up, free VRAM
        gc.collect()
        torch.cuda.empty_cache()

        instructed_prompt = f"[INST]<<SYS>>Explain the following Code Snippet in detail. <</SYS>>{prompt}[/INST]"
        tokens = self.model_wrapper.tokenizer(instructed_prompt, return_tensors="pt")
        tokens.to(self.model_wrapper.device)

        torch.cuda.synchronize()
        generation_start_time = time.time()

        self.model_wrapper.model.generate(**tokens, max_length=generate_k + len(tokens['input_ids'][0]))

        torch.cuda.synchronize()
        generation_stop_time = time.time()

        # Clean up, free VRAM
        del tokens
        gc.collect()
        torch.cuda.empty_cache()

        # Clear Intervention-Hooks and restore original Model (Pre-Transformation)
        self.model_wrapper.clear_hooks()
        self.restore_original_model()

        intervention_elapsed_time = intervention_stop_time - intervention_start_time
        generation_elapsed_time = generation_stop_time - generation_start_time

        return intervention_elapsed_time, generation_elapsed_time


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, help="specify the config file"
)
commandline_args = parser.parse_args()
args = pyhocon.ConfigFactory.from_dict(json.loads(_jsonnet.evaluate_file(commandline_args.config_path)))

model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model_wrapper = CodeLlamaModel(model, tokenizer=tokenizer, device=args.device)

intervention_controller = InterventionGenerationControllerForTiming(model_wrapper)

# Load LMDebuggerInterventionForTiming
intervention_controller.register_method(LMDebuggerInterventionForTiming(
    model_wrapper,
    args
))

# Load SAEs
sae_layers = []
for config_path in tqdm(args.sae_paths, desc="Loading SAE-Instances"):
    method = SAEIntervention(
        model_wrapper,
        args,
        config_path
    )
    intervention_controller.register_method(method)
    sae_layers.append(method.supported_layers[0])

# Load ROME Instances
rome_layers = []
for config_path in tqdm(args.rome_paths, desc="Loading ROME-Instances"):
    method = ROMEIntervention(
        model_wrapper,
        args,
        config_path
    )
    intervention_controller.register_method(method)
    rome_layers.append(method.supported_layers[0])

# Load The Stack dedup
ds_stack = load_dataset("bigcode/the-stack-dedup", streaming=True, split="train")
ds_stack_iterator = iter(ds_stack)

# Load CounterFact-Dataset by Meng et al.
ds_counterfact = load_dataset("azhx/counterfact")
ds_counterfact_iterator = iter(ds_counterfact["test"])

def calculate_throughput(gen_tokens, init_prompt_len=32):
    sample = next(ds_stack_iterator)["content"]
    tokens = tokenizer(sample, return_tensors="pt", truncation=True,
                       max_length=init_prompt_len, add_special_tokens=False)

    while len(tokens["input_ids"][0]) < init_prompt_len:
        sample = next(ds_stack_iterator)["content"]
        tokens = tokenizer(sample, return_tensors="pt", truncation=True,
                           max_length=init_prompt_len, add_special_tokens=False)

    intervention_elapsed_time, generation_elapsed_time = intervention_controller.generate(sample, gen_tokens)

    return len(intervention_controller.interventions) / intervention_elapsed_time, gen_tokens / generation_elapsed_time

if "throughput_plots" not in os.listdir("./"):
    os.mkdir("throughput_plots")

"""
Token Throughput relative to Amount of Interventions
"""

fig_token_throughput = go.Figure(layout={
    "title": "Throughput of Text Generation",
    "xaxis_title": "Number of Interventions",
    "yaxis_title": "Token Throughput [Tokens/s]",
    "font": dict(size=32)
})

fig_intervention_throughput = go.Figure(layout={
    "title": "Throughput of Preprocessing Interventions",
    "xaxis_title": "Number of Interventions",
    "yaxis_title": "Intervention Throughput [Interventions/s]",
    "font": dict(size=32)
})

tokens_to_generate = 500
num_interventions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
sae_dict_vec_size = 4 * 11008

baseline_token_throughput = []
baseline_intervention_throughput = []
lmd_token_throughput = []
lmd_intervention_throughput = []
sae_token_throughput = []
sae_intervention_throughput = []
rome_token_throughput = []
rome_intervention_throughput = []

# Baseline
for amount_interventions in tqdm([0], desc="Baseline"):
    intervention_controller.clear_interventions()

    intervention_throughput, token_throughput = calculate_throughput(tokens_to_generate)

    baseline_token_throughput = [token_throughput for _ in range(len(num_interventions))]

fig_token_throughput.add_trace(go.Scatter(x=num_interventions, y=baseline_token_throughput, name="Baseline"))

# LMDebuggerIntervention
for amount_interventions in tqdm(num_interventions, desc="LMDebuggerIntervention"):
    intervention_controller.clear_interventions()
    intervention_controller.set_interventions([{
        "type": "LMDebuggerInterventionForTiming",
        "dim": random.randint(0, model_wrapper.model.config.intermediate_size - 1),
        "layer": random.randint(0, model_wrapper.model.config.num_hidden_layers - 1),
        "coeff": random.randint(1, 100)
    } for _ in range(amount_interventions)])

    intervention_throughput, token_throughput = calculate_throughput(tokens_to_generate)

    lmd_token_throughput.append(token_throughput)
    lmd_intervention_throughput.append(intervention_throughput)

fig_token_throughput.add_trace(go.Scatter(x=num_interventions, y=lmd_token_throughput, name="LMDebuggerIntervention"))
fig_intervention_throughput.add_trace(go.Scatter(x=num_interventions,
                                                 y=lmd_intervention_throughput, name="LMDebuggerIntervention"))

# SAEIntervention
for amount_interventions in tqdm(num_interventions, desc="SAEIntervention"):
    intervention_controller.clear_interventions()
    intervention_controller.set_interventions([{
        "type": "SAEIntervention",
        "dim": random.randint(0, sae_dict_vec_size - 1),
        "layer": random.choice(sae_layers),
        "coeff": random.randint(0, 100)
    } for _ in range(amount_interventions)])

    intervention_throughput, token_throughput = calculate_throughput(tokens_to_generate)

    sae_token_throughput.append(token_throughput)
    sae_intervention_throughput.append(intervention_throughput)

fig_token_throughput.add_trace(go.Scatter(x=num_interventions, y=sae_token_throughput, name="SAEIntervention"))
fig_intervention_throughput.add_trace(go.Scatter(x=num_interventions,
                                                 y=sae_intervention_throughput, name="SAEIntervention"))

# ROMEIntervention
for amount_interventions in tqdm(num_interventions, desc="ROMEIntervention"):
    rome_interventions = []

    for _ in range(amount_interventions):
        ds_item = next(ds_counterfact_iterator)

        rome_interventions.append({
            "type": "ROMEIntervention",
            "layer": random.choice(rome_layers),
            "text_inputs": {
                "prompt": ds_item["requested_rewrite"]["prompt"],
                "subject": ds_item["requested_rewrite"]["subject"],
                "target": ds_item["requested_rewrite"]["target_new"]["str"]
            },
            "coeff": 1.0
        })

    intervention_controller.clear_interventions()
    intervention_controller.set_interventions(rome_interventions)

    intervention_throughput, token_throughput = calculate_throughput(tokens_to_generate)

    rome_token_throughput.append(token_throughput)
    rome_intervention_throughput.append(intervention_throughput)

fig_token_throughput.add_trace(go.Scatter(x=num_interventions, y=rome_token_throughput, name="ROMEIntervention"))
fig_intervention_throughput.add_trace(go.Scatter(x=num_interventions,
                                                 y=rome_intervention_throughput, name="ROMEIntervention"))

fig_token_throughput.write_html("throughput_plots/token_throughput_over_interventions.html")
fig_intervention_throughput.write_html("throughput_plots/intervention_throughput_over_interventions.html")
