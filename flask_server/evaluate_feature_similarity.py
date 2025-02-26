import torch
import pickle

import plotly.express as px

from sparse_autoencoders import CodeLlamaModel
from sparse_autoencoders.AutoEncoder import AutoEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "codellama/CodeLlama-7b-Instruct-hf"

sae_config_paths = [
    "/nfs/data/students/ebenz_bsc2024/autoenc_2/autoenc_lr2e-4_0.5_32_nr/5000.pt",
    "/nfs/data/students/ebenz_bsc2024/autoenc_2/autoenc_lr2e-4_0.5_32_nr/10000.pt",
    "/nfs/data/students/ebenz_bsc2024/autoenc_2/autoenc_lr2e-4_0.5_32_nr/20000.pt",
    "/nfs/data/students/ebenz_bsc2024/autoenc_2/autoenc_lr2e-4_0.5_32_nr/30000.pt",
    "/nfs/data/students/ebenz_bsc2024/autoenc_2/autoenc_lr2e-4_0.5_32_nr/40000.pt",
    "/nfs/data/students/ebenz_bsc2024/autoenc_2/autoenc_lr2e-4_0.5_32_nr/50000.pt"
]

device = "cuda:2"

"""
FUNCTIONS
"""
def calculate_max_similarities(a, b):
    # Normalize a and b
    feature_matrix_a = (a / torch.norm(a, p=2, dim=0, keepdim=True)).to(torch.float16).to(device)
    feature_matrix_b = (b / torch.norm(b, p=2, dim=0, keepdim=True)).to(torch.float16).to(device)

    full_similarity_matrix = feature_matrix_a.T @ feature_matrix_b
    max_similarity_matrix = torch.max(full_similarity_matrix, dim=0).values.detach().cpu()

    return max_similarity_matrix

# Load Transformer Model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_wrapper = CodeLlamaModel(model, tokenizer=tokenizer, device="cpu")

sae_names = []
similarities = []

for sae_config_path in sae_config_paths:
    # Load SAE
    with open(sae_config_path, "rb") as f:
        sae_config = pickle.load(f)
        sae_layer = sae_config["LAYER_INDEX"]

    autoencoder = AutoEncoder.load_model_from_config(sae_config)
    autoencoder = autoencoder.cpu()

    # Calculate max cosine-similarities
    a = model_wrapper.model.model.layers[sae_config["LAYER_INDEX"]].mlp.up_proj.weight
    b = autoencoder.weight_encoder.data

    max_similarities = calculate_max_similarities(a, b)

    # Plot histogram
    fig = px.histogram(x=max_similarities)
    fig.write_html(f"similarity_plots/{sae_config_path.split('/')[-1]}.html")

    mean_max_similarity = torch.mean(max_similarities).item()

    try:
        sae_names.append(int(sae_config_path.split('/')[-1].split('.')[0]))
    except ValueError:
        sae_names.append(sae_config_path.split('/')[-1].split('.')[0])
    similarities.append(mean_max_similarity)

fig = px.line(x=sae_names, y=similarities)
fig.write_html("similarity_plots/line.html")