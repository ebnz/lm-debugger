import os
import torch
import pickle
from tqdm import tqdm
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

from sparse_autoencoders import CodeLlamaModel
from sparse_autoencoders.AutoEncoder import AutoEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "codellama/CodeLlama-7b-Instruct-hf"

sae_config_paths = [
    "/nfs/data/students/ebenz_bsc2024/autoenc_layers/autoenc_l4",
    "/nfs/data/students/ebenz_bsc2024/autoenc_layers/autoenc_l12",
    "/nfs/data/students/ebenz_bsc2024/autoenc_layers/autoenc_l20",
    "/nfs/data/students/ebenz_bsc2024/autoenc_layers/autoenc_l28",
    "/nfs/data/students/ebenz_bsc2024/autoenc_layers/autoenc_sublayer_l4",
    "/nfs/data/students/ebenz_bsc2024/autoenc_layers/autoenc_sublayer_l12",
    "/nfs/data/students/ebenz_bsc2024/autoenc_layers/autoenc_sublayer_l20",
    "/nfs/data/students/ebenz_bsc2024/autoenc_layers/autoenc_sublayer_l28"
]

device = "cuda:3"

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

# For each Path, that contains multiple SAE-Models
for multi_sae_config_path in tqdm(sae_config_paths, desc="Directory No."):
    sae_dirlist = os.listdir(multi_sae_config_path)
    sae_name = multi_sae_config_path.split("/")[-1]

    # Datalines for Line Plot of Mean/Top10% Cos-Similarities of Training Progress
    sae_names = []
    mean_similarities = []
    top10_similarities = []

    # For each SAE-Model in one of the Paths
    for sae_config_path in tqdm(sae_dirlist, desc="SAE No.", leave=False):
        # Skip directories, files that are no models
        if not sae_config_path.endswith(".pt"):
            continue

        # Load SAE
        with open(f"{multi_sae_config_path}/{sae_config_path}", "rb") as f:
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

        if sae_name not in os.listdir("similarity_plots"):
            os.mkdir(f"similarity_plots/{sae_name}")

        fig.write_html(f"similarity_plots/{sae_name}/{sae_config_path.split('/')[-1]}.html")

        """
        Collect SAE-Name (commonly equals the number of training steps) and Mean Cos-Similarity
        Used in Step: Plot Mean Cos-Similarities over Training Progress
        """
        mean_max_similarity = torch.mean(max_similarities).item()
        mean_similarities.append(mean_max_similarity)

        top10_similarity = torch.quantile(max_similarities.to(torch.float32), q=0.9).item()
        top10_similarities.append(top10_similarity)

        try:
            sae_names.append(int(sae_config_path.split('/')[-1].split('.')[0]))
        except ValueError:
            sae_names.append(sae_config_path.split('/')[-1].split('.')[0])


    """
    Plot Mean/Top10% Cos-Similarities over Training Progress
    """
    # Sort x-y-Pairs
    mean_df = pd.DataFrame(data={"x": sae_names, "y": mean_similarities})
    mean_df.sort_values(by="x", inplace=True)
    fig_line_mean.add_trace(go.Scatter(
        x=mean_df["x"],
        y=mean_df["y"],
        name=f"{sae_name}"
    ))

    top10_df = pd.DataFrame(data={"x": sae_names, "y": top10_similarities})
    top10_df.sort_values(by="x", inplace=True)
    fig_line_top10.add_trace(go.Scatter(
        x=top10_df["x"],
        y=top10_df["y"],
        name=f"{sae_name}"
    ))

fig_line_mean.write_html(f"similarity_plots/line_mean.html")
fig_line_top10.write_html(f"similarity_plots/line_top10.html")

