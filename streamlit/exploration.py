import argparse
import json
import _jsonnet
import pickle

import streamlit as st
from elasticsearch import Elasticsearch

import pandas as pd

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="../config_files/codellama-7b-instruct.jsonnet")
args = parser.parse_args()
json_config = args.config_path
config = json.loads(_jsonnet.evaluate_file(json_config))

# Streamlit Configuration
st.set_page_config(layout="wide")

# Connect to ElasticSearch-Index
with Elasticsearch(
    f"{config['elastic_ip']}:{config['elastic_port']}",
    api_key=config["elastic_api_key"]
) as es_client:

    # Load Search-Indices
    search_indices = [config["elastic_index"]]
    search_indices_types = ["LMDebuggerIntervention"]

    for sae_path in config['sae_paths']:
        with open(sae_path, "rb") as f:
            sae_config = pickle.load(f)

        # ElasticSearch Index-Name
        index_name = (f'{sae_config["MODEL_TYPE"]}_{sae_config["LAYER_TYPE"]}_{sae_config["LAYER_INDEX"]}_'
                      f'{sae_config["ACT_VEC_SIZE"]}_{sae_config["DICT_VEC_SIZE"]}_{sae_config["LEARNING_RATE"]}_'
                      f'{sae_config["L1_COEFFICIENT"]}').lower()

        if bool(es_client.indices.exists(index=index_name)):
            search_indices.append(index_name)
            search_indices_types.append("SAEIntervention")

    keywords = st.text_input(label="Keywords", placeholder="List keywords to search for in comma-separated style")

    dict_for_df = {
        "index_type": [],
        "layer": [],
        "dim": [],
        "tokens": [],
        "search_score": [],
        "interpretability_score": [],
        "interpretation": []
    }
    for es_index, es_index_type in zip(search_indices, search_indices_types):
        result = es_client.search(index=es_index, q=keywords, size=config["top_k_tokens_for_ui"])["hits"]["hits"]

        for item in result:
            if es_index_type == "LMDebuggerIntervention":
                dict_for_df["index_type"].append(es_index_type)
                dict_for_df["layer"].append(item["_source"]["layer"])
                dict_for_df["dim"].append(item["_source"]["dim"])
                dict_for_df["tokens"].append(item["_source"]["tokens"])
                dict_for_df["search_score"].append(item["_score"])
                dict_for_df["interpretability_score"].append(None)
                dict_for_df["interpretation"].append(None)
            elif es_index_type == "SAEIntervention":
                dict_for_df["index_type"].append(es_index_type)
                dict_for_df["layer"].append(item["_source"]["layer"])
                dict_for_df["dim"].append(item["_source"]["dim"])
                dict_for_df["tokens"].append(item["_source"]["tokens"])
                dict_for_df["search_score"].append(item["_score"])
                dict_for_df["interpretability_score"].append(item["_source"]["score"])
                dict_for_df["interpretation"].append(item["_source"]["interpretation"])

    if len(dict_for_df["search_score"]) > 0:
        pandas_df = pd.DataFrame(data=dict_for_df)
        streamlit_df = st.dataframe(
            pandas_df,
            column_config={
                "index_type": "Type of Intervention Method",
                "layer": "Layer Index",
                "dim": "Feature Index",
                "tokens": "Strongly activating Tokens",
                "search_score": "Database-Hit strength",
                "interpretability_score": "Interpretability Score",
                "interpretation": st.column_config.Column(label="Interpretation", width=800)
            },
            hide_index=True
        )

    else:
        st.write("No Results :/")
