from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer
from math import isnan

from . import repr_tools
from ..util import nethook
from ..util.nethook import get_module

from .rome_hparams import ROMEHyperParams


def rebatch(input_tok, bs):
    n = len(next(iter(input_tok.values())))
    return [{k: v[idx: min(idx + bs, n)]
            for k, v in input_tok.items()} for idx in range(0, n, bs)]


def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    k_star,
    context_templates: List[str],
    verbose_output: bool = False,
    per_device_batch_size = 1,  # Conservative estimate to ensure minimal memory consumption.
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    device = next(model.parameters()).device

    print("Computing right vector (v)")

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [
        # prompt in different contexts including target
        context.replace("{"*2, "{"*4).replace("}"*2, "}"*4).format(request["prompt"]) + request["target_new"]["str"].replace("{", "{"*2).replace("}", "}"*2)
        for context in context_templates
    ]
    kl_prompts = ["{} is a"]

    # find target ids and remove last token from each rewriting prompt
    target_ids = []
    lookup_idxs = []
    for i, prompt in enumerate(rewriting_prompts):
        lookup_idxs.append(find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i==0)
        ))
        tokenized = tok(prompt.format(request["subject"]), return_tensors="pt").input_ids[0]
        target_ids.append(find_target_ids_at_end(tokenized, request["target_new"]["str"], tok))
        rewriting_prompts[i] = tok.decode(tokenized[:-1])
    
    kl_prompts = [prompt.format(request["subject"]) for prompt in kl_prompts]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        all_prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        if tok.padding_side == "right":
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids[i]) : ex_len] = target_ids[i]
        elif tok.padding_side == "left":
            rewriting_targets[i, -len(target_ids[i]):] = target_ids[i]
        else:
            raise ValueError(f"Unknown padding side {tok.padding_side}")

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    n_embd = None
    if hasattr(model.config, "n_embd"):
        n_embd = model.config.n_embd
    elif hasattr(model.config, "hidden_size"):
        n_embd = model.config.hidden_size
    else:
        assert False, "No hidden dimension found in config"
    delta = torch.zeros((n_embd,), requires_grad=True, device=device)
    target_init, kl_distr_init = None, None
    exec_count = 0  # I hate everything about this
    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        # This thing induces statefullness all over this entire implementation. Argghh
        nonlocal target_init, exec_count

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs[exec_count: exec_count + len(cur_out)]):
                cur_out[i, idx, :] += delta.to(cur_out.device)
            exec_count += len(cur_out)
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        exec_count = 0

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = torch.concat([model(**batch).logits for batch in rebatch(input_tok, per_device_batch_size)])

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / (sum(len(l) for l in target_ids) / len(target_ids))
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init.to(device))
        ) ** 2
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm().to(device)
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta.to(target_init.device)

    cur_output = get_module(model, hparams.rewrite_module_tmp.format(layer))(k_star)

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(k_star, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(k_star, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    if verbose_output:
        return right_vector, cur_output, target

    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found | Sentence: {sentence} | Subject: {subject} | Token:",
            tok.convert_ids_to_tokens([tok(sentence)["input_ids"][ret]])[0],
        )

    return ret


def find_target_ids_at_end(input_ids: str, target: str, tok: AutoTokenizer):
    for i in range(1, len(input_ids)):
        detokenized = tok.decode(input_ids[-i:])
        if len(detokenized) >= len(target) and detokenized[-len(target):] == target:
            return input_ids[-i:]
    raise ValueError(f"Did not find target '{target}' in {input_ids} ({tok.decode(input_ids)})")