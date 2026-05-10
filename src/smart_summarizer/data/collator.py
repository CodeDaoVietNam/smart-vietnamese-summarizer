from __future__ import annotations


def build_seq2seq_collator(tokenizer, model=None):
    from transformers import DataCollatorForSeq2Seq

    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
