import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import repr_tools
from ..util import nethook
from ..util.globals import *
from ..util.nethook import Trace

from .layer_stats import layer_stats
from .rome_hparams import ROMEHyperParams

# Cache variables
inv_mom2_cache = {}


def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")

    # Key cache of a) the results of a single forward pass up to the target layer.
    # And b) The number of samples used to estimate the second moment
    # Not exactly cryptographically secure but probably good enough for our use case.
    with Trace(
            model, layer_name, retain_input=True, retain_output=False, stop=True
    ) as tr:
        model(input_ids=torch.tensor([[42]], device=next(model.parameters()).device))


    dim = max(nethook.get_parameter(
        model, f"{layer_name}.weight"
    ).shape)

    if dim > mom2_n_samples:
        print(
            f"mom2_n_samples needs to be larger than layer dim to ensure invertibility of covariance matrix."
            f"Continuing with minimal viable number: mom2_n_samples = dim + 1 = {dim + 1}."
        )
        mom2_n_samples = dim + 1


    # transform input to make it better suited for use as file name later on
    key = str(tr.input.sum().abs().detach().cpu().numpy().item() + mom2_n_samples / 1_000).replace(".", "")

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )

        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            cache_key=key,
        )
        dtype = getattr(torch, mom2_dtype)
        # convert to 32 bit float
        # only 32/64 bit floats supported by inverse
        mom2 = stat.mom2.moment().float().to(model.device)

        inv_mom2_cache[key] = torch.inverse(mom2).type(dtype)
        
        print("got inverse covariance matrix")

    return inv_mom2_cache[key]


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.replace("{{", "{{{{").replace("}}", "}}}}").format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    u = cur_repr
    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        ).to(cur_repr.device, dtype=cur_repr.dtype) @ cur_repr.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm(), cur_repr
