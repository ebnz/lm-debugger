import argparse
import json
import os
import pickle
import warnings

import _jsonnet
import numpy as np
import pyhocon
import torch
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import LlamaForCausalLM, CodeLlamaTokenizer

warnings.filterwarnings('ignore')

def get_all_projected_values(model):
    logits = []
    for i in tqdm(range(model.config.num_hidden_layers)):
        layer_logits = torch.matmul(model.model.embed_tokens.weight, model.model.layers[i].mlp.down_proj.weight).T
        logits.append(layer_logits.detach().cpu())

    logits = torch.vstack(logits)
    return logits.numpy()


def create_elastic_search_data(path, model, model_name, tokenizer, top_k):
    if os.path.isfile(path):
        with open(path, 'rb') as handle:
            dict_es = pickle.load(handle)
            return dict_es
    d = {}
    inv_d = {}
    cnt = 0
    total_dims = model.model.layers[0].mlp.down_proj.weight.size(1)
    for i in range(model.config.num_hidden_layers):
        for j in range(total_dims):
            d[cnt] = (i, j)
            inv_d[(i, j)] = cnt
            cnt += 1
    dict_es = {}
    logits = get_all_projected_values(model)
    for i in tqdm(range(model.config.num_hidden_layers)):
        for j in tqdm(range(total_dims), leave=False):
            k = (i, j)
            cnt = inv_d[(i, j)]
            ids = np.argsort(-logits[cnt])[:top_k]
            ids_list = ids.tolist()     #Converting a copy to a python list, as tokenizer._convert_id_to_token has problems with np.array
            tokens = [tokenizer._convert_id_to_token(x) for x in ids_list]
            dict_es[k] = [(ids[b], tokens[b], logits[cnt][ids[b]]) for b in range(len(tokens))]
    with open(path, 'wb') as handle:
        pickle.dump(dict_es, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dict_es

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", default='./config_files/codellama-7b-instruct.jsonnet', type=str, help="specify the config file"
    )
    args = parser.parse_args()
    config = pyhocon.ConfigFactory.from_dict(json.loads(_jsonnet.evaluate_file(args.config_path)))
    model = LlamaForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float16)
    device = config.device
    model.to(device)
    tokenizer = CodeLlamaTokenizer.from_pretrained(config.model_name)
    _ = create_elastic_search_data(config.elastic_projections_path, model, config.model_name, tokenizer,
                                   config.top_k_for_elastic)
