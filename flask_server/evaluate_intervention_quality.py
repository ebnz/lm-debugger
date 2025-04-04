import os
import pickle

import torch
import json
import _jsonnet
import pyhocon
import argparse
import numpy as np

from elasticsearch import Elasticsearch
from tqdm import tqdm

import plotly.graph_objs as go

from sparse_autoencoders import CodeLlamaModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from intervention_methods.InterventionGenerationController import InterventionGenerationController
from intervention_methods.LMDebuggerIntervention import LMDebuggerIntervention
from intervention_methods.ROMEIntervention import ROMEIntervention
from intervention_methods.SAEIntervention import SAEIntervention


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, help="specify the config file"
)

parser.add_argument(
    "--interpretation_samples_path", type=str, help="specify the interpretation samples file"
)

parser.add_argument(
    "--layer_id", type=int, help="Layer to use for all Intervention Methods"
)

parser.add_argument(
    "--num_lmdeb_features", type=int, help="Number of LMDebuggerIntervention-Features to use"
)

parser.add_argument(
    "--num_sae_features", type=int, help="Number of SAEIntervention-Features to use"
)

parser.add_argument(
    "num_ds_samples", type=int, help="Number of Dataset Samples to process"
)

commandline_args = parser.parse_args()

args = pyhocon.ConfigFactory.from_dict(json.loads(_jsonnet.evaluate_file(commandline_args.config_path)))
INTERP_SAMPLES_PATH = commandline_args.interpretation_samples_path
LAYER_ID = commandline_args.layer_id
NUM_LMDEB_FEATURES = commandline_args.num_lmdeb_features
NUM_SAE_FEATURES = commandline_args.num_sae_features
NUM_DS_SAMPLES = commandline_args.num_ds_samples

model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model_wrapper = CodeLlamaModel(model, tokenizer=tokenizer, device=args.device)

intervention_controller = InterventionGenerationController(model_wrapper)

# Load LMDebuggerIntervention
intervention_controller.register_method(LMDebuggerIntervention(
    model_wrapper,
    args
))

# Load SAEs
used_sae = SAEIntervention(
        model_wrapper,
        args,
        args.sae_paths[0]
    )
intervention_controller.register_method(used_sae)

# Load ROME Instances
used_rome = ROMEIntervention(
        model_wrapper,
        args,
        args.rome_paths[0]
    )
intervention_controller.register_method(used_rome)

if not os.path.isdir("rqs"):
    os.mkdir("rqs")

LMDEB_INDEX_NAME = args["elastic_index"]
SAE_INDEX_NAME = "llama_sae"

# with open(INTERP_SAMPLES_PATH, "rb") as f:
#     obj = pickle.load(f)
#
# INTERPRETABLE_NEURON_INDICES = obj["interpretable_neuron_indices"].tolist()
# del obj

es_client = Elasticsearch(f"{args['elastic_ip']}:{args['elastic_port']}", api_key=args["elastic_api_key"])

if not es_client.indices.exists(index=SAE_INDEX_NAME):
    # Generate SAE_Index
    for dim in tqdm(range(used_sae.config.m), desc="Building ES-Index for SAE"):
        # Compute AutoEncoder-Output and set given Feature to high value
        f = torch.zeros(used_sae.autoencoder.m)
        f[dim] = args.sae_active_coeff

        # Calculate the output of the Decoder of the AutoEncoder
        x_hat = used_sae.autoencoder.forward_decoder(f.to(args.autoencoder_device)).detach().cpu().to(dtype=torch.float16)

        # Calculate the Output of the MLP-Block of the Model
        layer_type = used_sae.config["LAYER_TYPE"]
        layer_id = used_sae.config["LAYER_INDEX"]
        if layer_type == "mlp_activations":
            res_stream = model_wrapper.model.model.layers[layer_id].mlp.down_proj(
                x_hat.to(model_wrapper.device)
            )
        elif layer_type == "attn_sublayer":
            res_stream = x_hat.to(model_wrapper.device)
        elif layer_type == "mlp_sublayer":
            res_stream = x_hat.to(model_wrapper.device)
        else:
            raise AttributeError(f"layer_type <{layer_type}> unknown")

        # Calculate the Output-Logits of the Model and select those with highest probability
        normed_res_stream = model_wrapper.model.model.norm(res_stream)
        logits = model_wrapper.model.lm_head(normed_res_stream).detach().cpu()
        argsorted_logits = np.argsort(-1 * logits)[:args.top_k_for_elastic].tolist()

        # Logits and Tokens with highest probability
        output_logits = logits[argsorted_logits].tolist()
        output_tokens = [model_wrapper.tokenizer._convert_id_to_token(item) for item in argsorted_logits]

        document = {
            "layer": layer_id,
            "dim": dim,
            "tokens": output_tokens
        }

        es_client.create(document=document, index=SAE_INDEX_NAME, id=dim)

# Load CounterFact-Dataset by Meng et al.
ds_counterfact = load_dataset("azhx/counterfact")
ds_counterfact_iterator = iter(ds_counterfact["test"])

def efficacy(probs, true_token_id, new_token_id):
    efficacy_bool = True if probs[new_token_id] > probs[true_token_id] else False
    magnitude = probs[new_token_id] - probs[true_token_id]

    return efficacy_bool, magnitude

def specificity(probs, true_token_id, new_token_id):
    efficacy_bool = True if probs[new_token_id] < probs[true_token_id] else False
    magnitude = probs[true_token_id] - probs[new_token_id]

    return efficacy_bool, magnitude

def calculate_efficacy_from_model(tokens, true_token_id, new_token_id):
    model_output = model_wrapper.model(tokens.to(args.device))

    logits = model_output[0].detach().cpu()[0][-1]  # First (and only) Batch, last Token
    probs = torch.softmax(logits, dim=0)

    return efficacy(probs, true_token_id, new_token_id)

def calculate_specificity_from_model(tokens, true_token_id, new_token_id):
    model_output = model_wrapper.model(tokens.to(args.device))

    logits = model_output[0].detach().cpu()[0][-1]  # First (and only) Batch, last Token
    probs = torch.softmax(logits, dim=0)

    return specificity(probs, true_token_id, new_token_id)

baseline_efficacy_bools = []
baseline_efficacy_magnitudes = []
baseline_specificity_bools = []
baseline_specificity_magnitudes = []
baseline_generalization_bools = []
baseline_generalization_magnitudes = []

lmdebugger_efficacy_bools = []
lmdebugger_efficacy_magnitudes = []
lmdebugger_specificity_bools = []
lmdebugger_specificity_magnitudes = []
lmdebugger_generalization_bools = []
lmdebugger_generalization_magnitudes = []
lmdebugger_found_feature = []

sae_efficacy_bools = []
sae_efficacy_magnitudes = []
sae_specificity_bools = []
sae_specificity_magnitudes = []
sae_generalization_bools = []
sae_generalization_magnitudes = []
sae_found_feature = []

rome_efficacy_bools = []
rome_efficacy_magnitudes = []
rome_specificity_bools = []
rome_specificity_magnitudes = []
rome_generalization_bools = []
rome_generalization_magnitudes = []

for idx, ds_entry in tqdm(enumerate(ds_counterfact_iterator), desc="Dataset-Index", total=NUM_DS_SAMPLES):
    relation = ds_entry["requested_rewrite"]["prompt"]
    subject = ds_entry["requested_rewrite"]["subject"]
    target_true = ds_entry["requested_rewrite"]["target_true"]["str"]
    target_new = ds_entry["requested_rewrite"]["target_new"]["str"]

    paraphrase_prompts = ds_entry["paraphrase_prompts"]
    neighborhood_prompts = ds_entry["neighborhood_prompts"]

    """
    Baseline
    """
    with torch.no_grad():
        # Efficacy
        prompt = relation.format(subject)
        tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
        true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
        new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

        efficacy_bool, efficacy_magnitude = calculate_efficacy_from_model(tokens, true_token_id, new_token_id)

        baseline_efficacy_bools.append(efficacy_bool)
        baseline_efficacy_magnitudes.append(efficacy_magnitude)

        # Generalization
        for prompt in paraphrase_prompts:
            tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
            true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
            new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

            generalization_bool, generalization_magnitude = calculate_efficacy_from_model(tokens, true_token_id,
                                                                                          new_token_id)

            baseline_generalization_bools.append(generalization_bool)
            baseline_generalization_magnitudes.append(generalization_magnitude)

        # Specificity
        for prompt in neighborhood_prompts:
            tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
            true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
            new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

            specificity_bool, specificity_magnitude = calculate_specificity_from_model(tokens, true_token_id, new_token_id)

            baseline_specificity_bools.append(specificity_bool)
            baseline_specificity_magnitudes.append(specificity_magnitude)

    """
    LMDebuggerIntervention
    """
    # ES-Search for Features
    results = es_client.search(
        index=LMDEB_INDEX_NAME,
        size=NUM_LMDEB_FEATURES,
        query={
            "bool": {
                "must": [
                    {
                        "match": {
                            "tokens": target_new
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {
                            "layer": LAYER_ID
                        }
                    }
                ]
            }
        }
    )["hits"]["hits"]

    num_actually_used_features = min(NUM_LMDEB_FEATURES, len(results))
    feature_ids = [results[i]["_source"]["dim"] for i in range(num_actually_used_features)]
    lmdebugger_found_feature.append(num_actually_used_features)

    intervention_controller.set_interventions([
        {
            "type": "LMDebuggerIntervention",
            "dim": feature_id,
            "layer": LAYER_ID,
            "coeff": 1
        } for feature_id in feature_ids
    ])

    intervention_controller.setup_intervention_hooks("")

    with torch.no_grad():
        # Efficacy
        prompt = relation.format(subject)
        tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
        true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
        new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

        efficacy_bool, efficacy_magnitude = calculate_efficacy_from_model(tokens, true_token_id, new_token_id)

        lmdebugger_efficacy_bools.append(efficacy_bool)
        lmdebugger_efficacy_magnitudes.append(efficacy_magnitude)

        # Generalization
        for prompt in paraphrase_prompts:
            tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
            true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
            new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

            generalization_bool, generalization_magnitude = calculate_efficacy_from_model(tokens, true_token_id,
                                                                                          new_token_id)

            lmdebugger_generalization_bools.append(generalization_bool)
            lmdebugger_generalization_magnitudes.append(generalization_magnitude)

        # Specificity
        for prompt in neighborhood_prompts:
            tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
            true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
            new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

            specificity_bool, specificity_magnitude = calculate_specificity_from_model(tokens, true_token_id, new_token_id)

            lmdebugger_specificity_bools.append(specificity_bool)
            lmdebugger_specificity_magnitudes.append(specificity_magnitude)

    intervention_controller.model_wrapper.clear_hooks()

    """
    SAEIntervention
    """
    # ES-Search for Features
    results = es_client.search(
        index=SAE_INDEX_NAME,
        size=NUM_SAE_FEATURES,
        query={
            "bool": {
                "must": [
                    {
                        "match": {
                            "tokens": target_new
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {
                            "layer": LAYER_ID
                        }
                    }
                ]
            }
        }
    )["hits"]["hits"]

    num_actually_used_features = min(NUM_SAE_FEATURES, len(results))
    feature_ids = [results[i]["_source"]["dim"] for i in range(num_actually_used_features)]
    sae_found_feature.append(num_actually_used_features)

    intervention_controller.set_interventions([
        {
            "type": "SAEIntervention",
            "dim": feature_id,
            "layer": LAYER_ID,
            "coeff": 1      # Manipulated Feature coefficient is determined by "sae_active_coeff" in jsonnet config
        } for feature_id in feature_ids
    ])

    intervention_controller.setup_intervention_hooks("")

    with torch.no_grad():
        # Efficacy
        prompt = relation.format(subject)
        tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
        true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
        new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

        efficacy_bool, efficacy_magnitude = calculate_efficacy_from_model(tokens, true_token_id, new_token_id)

        sae_efficacy_bools.append(efficacy_bool)
        sae_efficacy_magnitudes.append(efficacy_magnitude)

        # Generalization
        for prompt in paraphrase_prompts:
            tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
            true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
            new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

            generalization_bool, generalization_magnitude = calculate_efficacy_from_model(tokens, true_token_id,
                                                                                          new_token_id)

            sae_generalization_bools.append(generalization_bool)
            sae_generalization_magnitudes.append(generalization_magnitude)

        # Specificity
        for prompt in neighborhood_prompts:
            tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
            true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
            new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

            specificity_bool, specificity_magnitude = calculate_specificity_from_model(tokens, true_token_id, new_token_id)

            sae_specificity_bools.append(specificity_bool)
            sae_specificity_magnitudes.append(specificity_magnitude)

    intervention_controller.model_wrapper.clear_hooks()

    if idx >= NUM_DS_SAMPLES:
        break

    """
    ROME
    """
    intervention_controller.set_interventions([
        {
            "type": "ROMEIntervention",
            "layer": LAYER_ID,
            "text_inputs": {
                "prompt": relation,
                "subject": subject,
                "target": target_new
            },
            "coeff": 1.0
        }
    ])

    intervention_controller.transform_model("")

    with torch.no_grad():
        # Efficacy
        prompt = relation.format(subject)
        tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
        true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
        new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

        efficacy_bool, efficacy_magnitude = calculate_efficacy_from_model(tokens, true_token_id, new_token_id)

        rome_efficacy_bools.append(efficacy_bool)
        rome_efficacy_magnitudes.append(efficacy_magnitude)

        # Generalization
        for prompt in paraphrase_prompts:
            tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
            true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
            new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

            generalization_bool, generalization_magnitude = calculate_efficacy_from_model(tokens, true_token_id,
                                                                                          new_token_id)

            rome_generalization_bools.append(generalization_bool)
            rome_generalization_magnitudes.append(generalization_magnitude)

        # Specificity
        for prompt in neighborhood_prompts:
            tokens = model_wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
            true_token_id = model_wrapper.tokenizer(target_true, add_special_tokens=False)["input_ids"][0]
            new_token_id = model_wrapper.tokenizer(target_new, add_special_tokens=False)["input_ids"][0]

            specificity_bool, specificity_magnitude = calculate_specificity_from_model(tokens, true_token_id, new_token_id)

            rome_specificity_bools.append(specificity_bool)
            rome_specificity_magnitudes.append(specificity_magnitude)

    intervention_controller.restore_original_model()

"""
Clean Up
"""
es_client.close()

"""
Calculate and Dump Metrics
"""
print("Calculating and Dumping Metrics")

def calculate_accuracy_from_bools(efficacy_bools):
    tensor_bools = torch.Tensor(efficacy_bools)
    return torch.nanmean(tensor_bools).item()

def calculate_mean_magnitude(magnitudes):
    tensor_magnitudes = torch.Tensor(magnitudes)
    return torch.nanmean(tensor_magnitudes).item()

with open("rqs/rq4_raw_data.pkl", "wb") as f:
    pickle.dump({
        "baseline_efficacy_bools": baseline_efficacy_bools,
        "baseline_efficacy_magnitudes": baseline_efficacy_magnitudes,
        "baseline_specificity_bools": baseline_specificity_bools,
        "baseline_specificity_magnitudes": baseline_specificity_magnitudes,
        "baseline_generalization_bools": baseline_generalization_bools,
        "baseline_generalization_magnitudes": baseline_generalization_magnitudes,

        "lmdebugger_efficacy_bools": lmdebugger_efficacy_bools,
        "lmdebugger_efficacy_magnitudes": lmdebugger_efficacy_magnitudes,
        "lmdebugger_specificity_bools": lmdebugger_specificity_bools,
        "lmdebugger_specificity_magnitudes": lmdebugger_specificity_magnitudes,
        "lmdebugger_generalization_bools": lmdebugger_generalization_bools,
        "lmdebugger_generalization_magnitudes": lmdebugger_generalization_magnitudes,

        "sae_efficacy_bools": sae_efficacy_bools,
        "sae_efficacy_magnitudes": sae_efficacy_magnitudes,
        "sae_specificity_bools": sae_specificity_bools,
        "sae_specificity_magnitudes": sae_specificity_magnitudes,
        "sae_generalization_bools": sae_generalization_bools,
        "sae_generalization_magnitudes": sae_generalization_magnitudes,

        "lmdebugger_found_feature": lmdebugger_found_feature,
        "sae_found_feature": sae_found_feature
    }, f)

with open("rqs/rq4_metrics.txt", "w") as f:
    f.write(f"baseline_efficacy_score: {calculate_accuracy_from_bools(baseline_efficacy_bools)}\n")
    f.write(f"baseline_efficacy_magnitude: {calculate_mean_magnitude(baseline_efficacy_magnitudes)}\n")

    f.write(f"baseline_specificity_score: {calculate_accuracy_from_bools(baseline_specificity_bools)}\n")
    f.write(f"baseline_specificity_magnitude: {calculate_mean_magnitude(baseline_specificity_magnitudes)}\n")

    f.write(f"baseline_generalization_score: {calculate_accuracy_from_bools(baseline_generalization_bools)}\n")
    f.write(f"baseline_generalization_magnitude: {calculate_mean_magnitude(baseline_generalization_magnitudes)}\n")

    f.write(f"lmdebugger_efficacy_score: {calculate_accuracy_from_bools(lmdebugger_efficacy_bools)}\n")
    f.write(f"lmdebugger_efficacy_magnitude: {calculate_mean_magnitude(lmdebugger_efficacy_magnitudes)}\n")

    f.write(f"lmdebugger_specificity_score: {calculate_accuracy_from_bools(lmdebugger_specificity_bools)}\n")
    f.write(f"lmdebugger_specificity_magnitude: {calculate_mean_magnitude(lmdebugger_specificity_magnitudes)}\n")

    f.write(f"lmdebugger_generalization_score: {calculate_accuracy_from_bools(lmdebugger_generalization_bools)}\n")
    f.write(f"lmdebugger_generalization_magnitude: {calculate_mean_magnitude(lmdebugger_generalization_magnitudes)}\n")

    f.write(f"sae_efficacy_score: {calculate_accuracy_from_bools(sae_efficacy_bools)}\n")
    f.write(f"sae_efficacy_magnitude: {calculate_mean_magnitude(sae_efficacy_magnitudes)}\n")

    f.write(f"sae_specificity_score: {calculate_accuracy_from_bools(sae_specificity_bools)}\n")
    f.write(f"sae_specificity_magnitude: {calculate_mean_magnitude(sae_specificity_magnitudes)}\n")

    f.write(f"sae_generalization_score: {calculate_accuracy_from_bools(sae_generalization_bools)}\n")
    f.write(f"sae_generalization_magnitude: {calculate_mean_magnitude(sae_generalization_magnitudes)}\n")

    f.write(f"rome_efficacy_score: {calculate_accuracy_from_bools(rome_efficacy_bools)}\n")
    f.write(f"rome_efficacy_magnitude: {calculate_mean_magnitude(rome_efficacy_magnitudes)}\n")

    f.write(f"rome_specificity_score: {calculate_accuracy_from_bools(rome_specificity_bools)}\n")
    f.write(f"rome_specificity_magnitude: {calculate_mean_magnitude(rome_specificity_magnitudes)}\n")

    f.write(f"rome_generalization_score: {calculate_accuracy_from_bools(rome_generalization_bools)}\n")
    f.write(f"rome_generalization_magnitude: {calculate_mean_magnitude(rome_generalization_magnitudes)}\n")

    f.write(f"ratio_lmdebugger_found_feature: {sum([1 if item > 0 else 0 for item in lmdebugger_found_feature]) / len(lmdebugger_found_feature)}\n")
    f.write(f"ratio_sae_found_feature: {sum([1 if item > 0 else 0 for item in sae_found_feature]) / len(sae_found_feature)}\n")
