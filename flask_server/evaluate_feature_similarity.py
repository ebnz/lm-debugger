import os
import torch
import pickle
from tqdm import tqdm
import pandas as pd
import argparse
import pyhocon
import json
import _jsonnet

import plotly.express as px
import plotly.graph_objs as go

from sparse_autoencoders import CodeLlamaModel
from sparse_autoencoders.AutoEncoder import AutoEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, help="specify the config file"
)
commandline_args = parser.parse_args()
args = pyhocon.ConfigFactory.from_dict(json.loads(_jsonnet.evaluate_file(commandline_args.config_path)))

model_name = args.model_name
device = args.device
sae_config_paths = args.sae_paths

if "similarity_plots" not in os.listdir("./"):
    os.mkdir("similarity_plots")

"""
FUNCTIONS
"""
def calculate_max_similarities(a, b):
    # Normalize a and b
    feature_matrix_a = (a / torch.norm(a, p=2, dim=0, keepdim=True))
    feature_matrix_b = (b / torch.norm(b, p=2, dim=0, keepdim=True))

    shape_a = feature_matrix_a.shape
    shape_b = feature_matrix_b.shape

    if shape_a[1] == shape_b[0]:
        full_similarity_matrix = feature_matrix_a @ feature_matrix_b
    elif shape_a[0] == shape_b[0]:
        full_similarity_matrix = feature_matrix_a.T @ feature_matrix_b
    elif shape_a[1] == shape_b[1]:
        full_similarity_matrix = feature_matrix_a @ feature_matrix_b.T
    elif shape_a[0] == shape_b[1]:
        full_similarity_matrix = feature_matrix_a.T @ feature_matrix_b.T
    else:
        raise ValueError(f"Shapes {shape_a} and {shape_b} are not compatible. Check Input Matrices")

    max_similarity_matrix = torch.max(full_similarity_matrix, dim=0).values

    return max_similarity_matrix

# Load Transformer Model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_wrapper = CodeLlamaModel(model, tokenizer=tokenizer, device="cpu")

# Figure for Lineplot of Mean/Top10% Cos-Similarities of Training Progress
fig_line_mean = go.Figure(layout={
    "xaxis_title": "Training Steps",
    "yaxis_title": "Mean Cos-Similarity",
    "title": "Mean Cos-Similarities over Training Progress",
    "font": dict(size=32)
})

fig_line_top10 = go.Figure(layout={
    "xaxis_title": "Training Steps",
    "yaxis_title": "Top-10% Cos-Similarity",
    "title": "Top-10% Quantile Cos-Similarities over Training Progress",
    "font": dict(size=32)
})

# For each SAE-Model
for sae_config_path in tqdm(sae_config_paths, desc="SAE No.", leave=False):
    # Load SAE
    with open(sae_config_path, "rb") as f:
        sae_config = pickle.load(f)
        sae_layer = sae_config["LAYER_INDEX"]

    autoencoder = AutoEncoder.load_model_from_config(sae_config)
    autoencoder = autoencoder.cpu()

    """
    Calculate Cosine-Similarities
    """
    a = model_wrapper.model.model.layers[sae_config["LAYER_INDEX"]].mlp.down_proj.weight
    b = autoencoder.weight_encoder.data

    a = a.to(torch.float16).to(device)
    b = b.to(torch.float16).to(device)

    max_similarities = calculate_max_similarities(a, b)

    max_similarities = max_similarities.detach().cpu()

    """
    Plot Histogram of Cos-Similarities of one SAE
    """
    fig = px.histogram(
        x=max_similarities,
        labels={
            "x": "Cos-Similarity",
            "y": "Count"
        },
        title="Histogram of Cos-Similarities of Features"
    )

    fig.update_layout({"font": dict(size=64)})
    fig.write_html(f"similarity_plots/{sae_config_path.split('/')[-1]}.html")
