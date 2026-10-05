# Architecture: Generator-Independent Steganographic Codebooks

This fork changes the central abstraction inherited from Conversation Stenography.

The upstream codec assigns payload bits from a token's **rank in one exact
language model's next-token distribution**. That makes the carrier generator
part of the decoding protocol: sender and receiver must reproduce the same
weights, tokenizer, filtering rules, runtime behavior, and next-token ordering.

This fork moves protocol meaning out of the generator.

## Core invariant

The carrier language model supplies **plausible candidates**. A shared keyed
codebook assigns **payload symbols** to visible tokens or semantic regions.

```text
plaintext
   |
compression / framing
   |
authenticated encryption
   |
payload symbols
   |
keyed codebook constraint <---- shared secret + canonical decoding profile
   |
carrier generator         <---- sender-local and replaceable
   |
ordinary-looking text
   |
transport
   |
canonical tokenizer / embedder
   |
keyed codebook
   |
payload symbols
   |
authenticated decryption
```

The receiver must not need the sender's causal-LM weights or next-token logits.
That requirement is an acceptance test, not an optimization target.

## Phase 1: keyed token partition

`PRFSymbolMap` is intentionally simple. It uses HMAC over the shared key,
visible token history, and candidate token ID to assign each token to a compact
symbol class. `KeyedTokenEncoder` scans the generator's ranked candidates and
selects the highest-ranked candidate that belongs to the required class.
`KeyedTokenDecoder` reconstructs the symbol stream from token IDs alone.

This phase proves the architectural separation. It is **not** presented as a
semantic-robustness solution or a new cryptographic primitive.

## Phase 2: semantic codebooks

Replace the bootstrap PRF partition with a representation-aware map while
keeping the same `SymbolMap` boundary. Candidate tokens or phrases should first
be grouped by semantic representation, then keyed/contextualized inside those
regions so that payload constraints do not systematically force implausible
semantic jumps.

Relevant prior art includes:

- SemStamp (NAACL 2024): partitions sentence embedding space with locality-
  sensitive hashing and rejection-samples generations into allowed semantic
  regions.
- PASA (ICML 2026): uses semantic clusters plus secret-key/history-synchronized
  randomness to couple generation and semantic keys.

Their published mechanisms are prior art and design input. Their repositories
currently do not declare a repository license, so this project should
reimplement the relevant ideas rather than copy source code unless licensing is
clarified.

## Phase 3: dual lanes / bounded deniability

A visible token can be constrained simultaneously under more than one keyed
map. Conceptually:

```text
Symbol(real_key, token, context)   == next_real_symbol
Symbol(decoy_key, token, context)  == next_decoy_symbol
```

Candidate generation then searches for a plausible token satisfying both
constraints. Each designated key can recover an independently authenticated
stream from the same visible carrier.

The cryptographic layer establishes whether an opening is valid. An LLM may be
used above that layer to render a pre-existing decoy state naturally, but model
plausibility must never substitute for authentication.

## Explicit non-goals for the first implementation

- perfect forensic deniability against arbitrary multi-snapshot adversaries;
- endpoint fine-tuning as a prerequisite for ordinary users;
- secrecy of the carrier model;
- treating embeddings or model obscurity as encryption;
- requiring sender and receiver to run the same causal language model;
- copying unlicensed research code.

## Security boundary

Authenticated encryption remains the confidentiality and integrity boundary.
The learned/codebook layers provide concealment, robustness, and optional
ambiguity. Compromise of those layers must not be treated as equivalent to a
cryptographic break.

## Immediate engineering acceptance tests

1. Two senders using different candidate rankings can produce different
   carriers that decode to the same payload.
2. A receiver can decode without loading a causal language model.
3. Carrier tokenization round-trips exactly under the canonical tokenizer.
4. Wrong or malformed streams fail framing/authentication once integrated with
   the conversation chain.
5. Future semantic maps can replace `PRFSymbolMap` without changing encryption
   or framing APIs.
