"""Local text generation for the explanation layer -- the model loader.

Mirrors scoring.py. One place owns HOW the model is loaded, because model
identity includes the quantisation and the device, not just the checkpoint
name: the same weights in NF4 and in fp16 are two different generators, and an
explanation must carry which one produced it.

LOCAL ONLY
----------
The golden set is MIMIC-derived and DUA-covered, so handing it to a hosted API
would redistribute credentialed data. Nothing in this module can reach a
network except the one-time weight download from the Hub.

GREEDY, SEEDED, NOT SAMPLED
---------------------------
do_sample=False and a fixed seed. This is not a quality choice -- it is what
makes a prompt change attributable. The whole point of grounding.py is to say
"this edit removed that violation", and a generator that answers differently on
the same input each run cannot support that claim.

WHAT THE GENERATOR IS
---------------------
Exactly the `generator(payload) -> str` callable that explain.explain() already
expects. Nothing here decides WHETHER a record may be explained; the
sufficiency gate lives upstream in explain.build_payload(), which raises rather
than returning something a generator could consume.

⚠️ transformers MUST BE < 5
---------------------------
transformers 5.14.1 SEGFAULTS (exit 139) loading a 7B in NF4 on this card. Not
an exception, a hard crash in native code, and it reproduces across device_map
"auto" and {"": 0}, with and without double quantisation, on both float16 and
bfloat16 compute. The same stack loads a 0.5B fine, so it presents as a size
problem and is easy to misattribute to bitsandbytes.

It is not bitsandbytes. On transformers 4.57.6 the identical 7B NF4 load
succeeds in 23 s at 5.57 GB VRAM, same bitsandbytes 0.50.0, same driver, same
weights. Pin below 5 until that is fixed upstream.

Two other Windows-specific notes from the same investigation:
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is a NO-OP here -- torch
    warns "not supported on this platform". It cannot be used to work around
    allocator fragmentation on Windows.
  * A failed load can surface as OSError 1455 "the paging file is too small"
    rather than as an OOM. That is commit exhaustion, not a GPU problem; check
    what else is holding RAM before touching the model configuration.
"""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field

from .. import config as C
from . import explain as E


def _quant_config(quant: str):
    """NF4 config, or None for fp16. `none` is the documented fallback for a
    smaller model if 4-bit ever stops working on this card."""
    if quant == "none":
        return None
    if quant != "nf4":
        raise ValueError(f"unknown LLM_QUANT {quant!r}; expected 'nf4' or 'none'")
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


@dataclass(slots=True)
class Generator:
    model: object
    tokenizer: object
    provenance: dict
    max_new_tokens: int = C.LLM_MAX_NEW_TOKENS
    latencies: list = field(default_factory=list)

    def __call__(self, payload: dict) -> str:
        import torch
        msgs = E.render_prompt(payload)
        enc = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        n_in = enc["input_ids"].shape[1]
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        self.latencies.append(time.time() - t0)
        # Only the completion. Decoding the whole sequence would hand the
        # verifier the system prompt as well, and the system prompt names every
        # band -- which would trip BAND_MISMATCH on text that never said it.
        return self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


def capability_check() -> dict:
    """The cheap check the stage runs every time: is the toolchain present and
    is there room?

    Deliberately does NOT load a model. Loading a probe model in the same
    process as the real one leaves a CUDA context and fragmented segments
    behind, and a 7B in NF4 needs a ~4.8 GB block on a card that only has ~6.4
    GB free once the display has taken its share -- the probe was the reason the
    real load then failed on fragmentation. The full probe lives in preflight(),
    which is a separate command.
    """
    import torch
    import huggingface_hub.constants as K

    # WHERE THE WEIGHTS WILL LAND, asserted rather than assumed.
    #
    # config.py sets HF_HOME, and that resolves correctly when tested. It did
    # not hold on 2026-08-08: 14.19 GB of Qwen weights appeared under
    # C:\Users\...\.cache\huggingface\hub, a full second copy beside the 15.23 GB
    # already on D:, and took the system drive to 1.23 GB free. The trigger was
    # never reproduced -- it coincided with huggingface_hub going 1.27.0 -> 0.36.2
    # on a transformers downgrade.
    #
    # An unreproducible cause is exactly the kind worth guarding rather than
    # explaining. Downloading 14 GB to the wrong drive should be a loud failure,
    # not something discovered when the disk fills.
    hub = pathlib.Path(K.HF_HUB_CACHE).resolve()
    if not str(hub).lower().startswith(str(pathlib.Path(C.LLM_CACHE).resolve()).lower()):
        raise RuntimeError(
            f"the model cache resolves to {hub}, not {C.LLM_CACHE}. Weights are "
            f"~15 GB and this would fill the system drive. Set HF_HOME (and "
            f"HF_HUB_CACHE) explicitly before importing transformers.")

    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    out = {"gpu": p.name, "compute_capability": f"sm_{p.major}{p.minor}",
           "vram_free_gb": round(free / 1e9, 2), "vram_total_gb": round(total / 1e9, 2),
           "hub_cache": str(hub),
           "quantisation": C.LLM_QUANT, "torch": torch.__version__}
    if C.LLM_QUANT != "none":
        import bitsandbytes
        out["bitsandbytes"] = bitsandbytes.__version__
    return out


def preflight(model_id: str | None = None) -> dict:
    """Prove the toolchain works before committing to a multi-GB download.

    bitsandbytes needs sm_120 kernels and Blackwell support is recent enough to
    be version-sensitive. A ~350 MB probe that quantises and generates one token
    costs seconds; discovering the same failure after pulling 5 GB costs an hour
    and tells you nothing extra.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = model_id or C.LLM_PREFLIGHT_MODEL
    p = torch.cuda.get_device_properties(0)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, quantization_config=_quant_config(C.LLM_QUANT), device_map={"": 0})
    model.eval()
    enc = tok.apply_chat_template([{"role": "user", "content": "Reply with exactly: ok"}],
                                  add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    vram = torch.cuda.memory_allocated() / 1e9
    del model
    torch.cuda.empty_cache()
    return {"probe_model": mid, "device": f"sm_{p.major}{p.minor}", "gpu": p.name,
            "quantisation": C.LLM_QUANT, "vram_gb": round(vram, 2),
            "seconds": round(time.time() - t0, 1), "output": txt}


def load_generator() -> Generator:
    """The model, the tokenizer, and the provenance that must travel with every
    line it writes."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(C.LLM_SEED)
    tok = AutoTokenizer.from_pretrained(C.LLM_MODEL_ID, revision=C.LLM_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        C.LLM_MODEL_ID, revision=C.LLM_REVISION,
        quantization_config=_quant_config(C.LLM_QUANT), device_map={"": 0})
    model.eval()
    gpu = torch.cuda.get_device_properties(0)

    prov = {
        "model": C.LLM_MODEL_ID,
        "revision": C.LLM_REVISION,
        "quantisation": C.LLM_QUANT,
        "dtype": str(model.dtype),
        "device": str(model.device),
        "compute_capability": f"sm_{gpu.major}{gpu.minor}",
        "max_new_tokens": C.LLM_MAX_NEW_TOKENS,
        "seed": C.LLM_SEED,
        "decoding": "greedy",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    if C.LLM_QUANT != "none":
        import bitsandbytes
        prov["bitsandbytes"] = bitsandbytes.__version__
    return Generator(model=model, tokenizer=tok, provenance=prov)
