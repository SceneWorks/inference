#!/usr/bin/env python3
"""Regenerate the two tiny Gemma-shaped tokenizer fixtures (sc-18762).

They are byte-for-byte real HF fast tokenizers — small enough to commit — that differ **only** in
the one field the LTX exactly-one-BOS policy turns on:

* ``tiny_gemma4_tokenizer.json`` — ``post_processor: null``, so ``encode(add_special_tokens=True)``
  emits **no** leading ``<bos>``. This is the shape measured on the Gemma 4 ``tokenizer.json``
  packed inside ``gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`` (a ``TemplateProcessing`` whose
  ``single`` is a bare ``$A`` with no special tokens).
* ``tiny_gemma3_tokenizer.json`` — a ``TemplateProcessing`` that prepends ``<bos>``, the shape
  measured on gemma-3-12b-it's ``tokenizer.json``.

Keeping both in CI means the policy's two failure modes (missing BOS / duplicate BOS) are covered
without a 32 MB vocabulary. Requires `pip install tokenizers`.
"""

from tokenizers import Tokenizer, models, pre_tokenizers, processors

VOCAB = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3,
    "a": 4,
    "red": 5,
    "fox": 6,
    "in": 7,
    "the": 8,
    "snow": 9,
    "café": 10,
    "日本語": 11,
}


def build(with_bos_post_processor: bool) -> Tokenizer:
    tk = Tokenizer(models.WordLevel(vocab=dict(VOCAB), unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    if with_bos_post_processor:
        tk.post_processor = processors.TemplateProcessing(
            single="<bos> $A", pair="<bos> $A <bos> $B:1", special_tokens=[("<bos>", 2)]
        )
    tk.add_special_tokens(["<pad>", "<eos>", "<bos>", "<unk>"])
    return tk


for name, flag in (("tiny_gemma4_tokenizer", False), ("tiny_gemma3_tokenizer", True)):
    tk = build(flag)
    with open(f"{name}.json", "w") as handle:
        handle.write(tk.to_str(pretty=False))
    print(name, [tk.encode(t, add_special_tokens=True).ids for t in ("", "a red fox in the snow")])
