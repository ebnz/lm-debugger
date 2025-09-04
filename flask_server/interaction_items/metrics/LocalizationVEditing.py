import torch
from .MetricItem import MetricItem

# From Notebook
import os, re, json
import numpy
from tqdm import tqdm
from collections import defaultdict
from .my_rome.util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset


class LocalizationVEditingMetric(MetricItem):
    def __init__(self, controller):
        super().__init__(controller)

    def get_text_outputs(self, token_logits, additional_params=None):


        return {
            "LocalizationVEditing": "Hallo"
        }