from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..util.logit_lens import LogitLens


def generate_interactive(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    top_k: int = 5,
    max_out_len: int = 200,
    compare_against: Optional[AutoModelForCausalLM] = None,
    use_logit_lens: bool = False,
    layer_module_tmp: str = "transformer.h.{}",
    ln_f_module: str = "transformer.ln_f",
    lm_head_module: str = "lm_head",
):
    """
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    if use_logit_lens:
        llens_gen = LogitLens(
            model,
            tok,
            layer_module_tmp,
            ln_f_module,
            lm_head_module,
            disabled=not use_logit_lens,
        )
        if compare_against:
            llens_vanilla = LogitLens(
                compare_against,
                tok,
                layer_module_tmp,
                ln_f_module,
                lm_head_module,
                disabled=not use_logit_lens,
            )

    while True:
        prompt = input("Enter a prompt: ").strip(" \r\t\n")

        print(
            f"Argument Model: "
            f"{generate_fast(model, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
        )
        if compare_against:
            print(
                f"Baseline Model: "
                f"{generate_fast(compare_against, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
            )

        if use_logit_lens:
            inp_prompt = tok([prompt], padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

            with llens_gen:
                model(**inp_prompt)
            print("\n--- Argument Model Logit Lens ---")
            llens_gen.pprint()

            if compare_against:
                with llens_vanilla:
                    compare_against(**inp_prompt)
                print("--- Baseline Model Logit Lens ---")
                llens_vanilla.pprint()

        print()


def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    verbose: bool = False,
    do_sample: bool = True,
    temperature: float = 1.,
):
    if isinstance(prompts, str):
        prompts = [prompts]
    if max_out_len is None and max_new_tokens is None:
        max_out_len = 200
    device = next(model.parameters()).device

    # batched generation only works correct with padding_side = left
    original_padding_side = tok.padding_side
    tok.padding_side = "left"
    
    try:
        inputs = tok(prompts, return_tensors="pt", padding=len(prompts) > 1, truncation=True, add_special_tokens=False).to(device)
    finally:
        # always reset padding side to ensure no change of state through generation
        tok.padding_side = original_padding_side


    out = model.generate(
        **inputs,
        top_k=top_k,
        temperature=temperature,
        do_sample=do_sample,
        num_return_sequences=n_gen_per_prompt,
        max_length=max_out_len,
        max_new_tokens=max_new_tokens,
    )
    if verbose:
        print("Inputs: ", inputs)
        print("Output tokens: ", out)
    return tok.batch_decode(out)