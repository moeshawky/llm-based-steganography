#!/usr/bin/env python3
"""Persistent deterministic Hugging Face backend for Conversation Stenography.

Install: python3 -m pip install torch transformers
Run indirectly through: Conversation Stenography generate ... / Conversation Stenography extract ...
"""

import argparse
import hashlib
import json
import sys


def reply(**values):
    print(json.dumps({"ok": True, **values}, separators=(",", ":")), flush=True)


def fail(exc):
    print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype)
    model.to(args.device)
    model.eval()
    # Exceeding a decoder-only context window does not raise: on CUDA it
    # usually surfaces as a device-side assert that kills the process, which
    # destroys every earlier record in the conversation. Refuse the request
    # up front so the caller sees the limit instead of a crash.
    limit = getattr(model.config, "max_position_embeddings", None)
    if not limit:
        limit = getattr(getattr(model.config, "rope_scaling", None) or {}, "original_max_position_embeddings", None)
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision
    identity = json.dumps({
        "model": args.model,
        "revision": resolved_revision,
        "tokenizer": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }, sort_keys=True).encode()
    fingerprint = "hf:" + hashlib.sha256(identity).hexdigest()
    special = set(tokenizer.all_special_ids)
    for token, token_id in tokenizer.get_vocab().items():
        if token.startswith("<|") and token.endswith("|>"):
            special.add(token_id)
    token_text_cache = {}
    # Prefix-matched KV cache. Every "next" request arrives with the full context,
    # but during carrier generation that context grows by exactly one token per
    # request, so a full re-forward per token dominates the runtime. Reuse is only
    # permitted when the incoming context starts with the tokens already held, so
    # a cache hit can only ever return the state a full forward would have built.
    cache = None
    cached_ids = []
    cached_logits = None

    def next_logits(ids):
        """Logits for the position after ids, reusing the prefix cache when exact."""
        nonlocal cache, cached_ids, cached_logits
        hold = len(cached_ids)
        if cache is not None and hold > 0 and ids[:hold] == cached_ids:
            if hold == len(ids):
                return cached_logits
            fresh = torch.tensor([ids[hold:]], device=args.device)
            with torch.inference_mode():
                out = model(input_ids=fresh, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            cached_ids = list(ids)
            cached_logits = out.logits[0, -1].float()
            return cached_logits
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=args.device), use_cache=True)
        cache = out.past_key_values
        cached_ids = list(ids)
        cached_logits = out.logits[0, -1].float()
        return cached_logits

    def token_text(token_id):
        if token_id not in token_text_cache:
            token_text_cache[token_id] = tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        return token_text_cache[token_id]

    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "info":
                reply(fingerprint=fingerprint)
            elif op == "tokenize":
                ids = tokenizer.encode(request.get("text", ""), add_special_tokens=False)
                reply(tokens=ids)
            elif op == "detokenize":
                text = tokenizer.decode(request.get("tokens", []), skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                reply(text=text)
            elif op == "next":
                ids = request["tokens"]
                top_n = int(request["top_n"])
                visible_tokens = request.get("visible_tokens")
                if not ids:
                    raise ValueError("model context cannot be empty")
                if limit and len(ids) > int(limit):
                    raise ValueError(
                        f"context of {len(ids)} tokens exceeds the model limit of {int(limit)}; "
                        "start a new conversation to reset the context"
                    )
                with torch.inference_mode():
                    # Clone so the special-token mask below never mutates the
                    # tensor retained by the prefix cache.
                    logits = next_logits(ids).clone()
                    # The mask must be applied inside inference_mode: newer torch
                    # versions reject in-place writes to inference tensors made
                    # from outside the block.
                    if special:
                        logits[list(special)] = -torch.inf
                pool_n = top_n if visible_tokens is None else min(logits.shape[-1], top_n * 4)
                values, indices = torch.topk(logits, k=pool_n, sorted=True)
                candidates = []
                tail = list(visible_tokens[-8:]) if visible_tokens else []
                for i, v in zip(indices.tolist(), values.tolist()):
                    i = int(i)
                    if visible_tokens is not None:
                        probe = tail + [i]
                        probe_text = tokenizer.decode(probe, skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)
                        encoded = tokenizer.encode(probe_text, add_special_tokens=False)
                        if encoded != probe:
                            continue
                    candidates.append({"id": i, "score": float(v), "text": token_text(i)})
                    if len(candidates) == top_n:
                        break
                if len(candidates) != top_n:
                    raise ValueError(f"only {len(candidates)} copy-safe candidates available; need {top_n}")
                reply(candidates=candidates)
            else:
                raise ValueError("unknown operation: " + str(op))
        except Exception as exc:
            fail(exc)


if __name__ == "__main__":
    main()
