"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..util import nethook


def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )


def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: List[str], words: List[str], subtoken: str
) -> List[int]:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), f"We require exactly one fill-in for context, {[tmp.count('{}') for tmp in context_templates]} were provided"

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    # adjust for doubled braces that will be removed by formatting
    fill_idxs = [idx - tmp[:idx].count("{{") - tmp[:idx].count("}}") for idx, tmp in zip(fill_idxs, context_templates)]
    prefixes = [
        tmp.format(words[i])[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ]

    tokenized = tok([tmp.format(word) for tmp, word in zip(context_templates, words)])["input_ids"]
    word_idxss = []
    for i in range(len(context_templates)):
        prefix = prefixes[i]
        word = words[i]
        prefix_with_start_of_word = prefix + word[0]
        prefix_with_full_word = prefix + word
        word_idxs = []
        for idx in range(len(tokenized[i])):
            detokenized = tok.decode(tokenized[i][:idx+1]).replace("<s> ", "<s>").replace("</s> ", "</s>")\
                .replace("<pad> .", "<pad>.")  # full stops preceded by padding get tokenized to the id for "whitespace full stop". This gets decoded to " .".  Yet another example of tokenization being a non-reversible action.
            if prefix_with_start_of_word in detokenized:
                word_idxs.append(idx)
            if prefix_with_full_word in detokenized:
                break
        if len(word_idxs) == 0:
            print("prefix")
            print(prefix)
            print("detokenized")
            print(detokenized)
        word_idxss.append(word_idxs)

    if subtoken == "last":
        return [[idxs[-1]] for idxs in word_idxss]
    elif subtoken == "first_after_last":
        return [[idxs[-1]+1] for idxs in word_idxss]
    elif subtoken == "first":
        return [[idxs[0]] for idxs in word_idxss]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for batch_contexts, batch_idxs in _batch(n=512):
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)
        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]