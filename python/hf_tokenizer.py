#!/usr/bin/env python3
"""Tokenizer-only backend for the keyed codebook receiver.

Unlike hf_model.py, this process never loads causal-LM weights. It exists to
make the architectural boundary explicit: carrier generation is sender-local;
decoding requires only the canonical tokenizer plus the shared SymbolMap.
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
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    resolved_revision = (
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or getattr(tokenizer, "_commit_hash", None)
        or args.revision
    )
    identity = json.dumps(
        {
            "tokenizer": args.tokenizer,
            "revision": resolved_revision,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "transformers": transformers.__version__,
        },
        sort_keys=True,
    ).encode()
    fingerprint = "hf-tokenizer:" + hashlib.sha256(identity).hexdigest()

    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "info":
                reply(fingerprint=fingerprint)
            elif op == "tokenize":
                reply(tokens=tokenizer.encode(request.get("text", ""), add_special_tokens=False))
            elif op == "detokenize":
                reply(
                    text=tokenizer.decode(
                        request.get("tokens", []),
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                )
            else:
                raise ValueError("unknown operation: " + str(op))
        except Exception as exc:
            fail(exc)


if __name__ == "__main__":
    main()
