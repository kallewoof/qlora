# Overview

This is a fork of [QLoRA](https://github.com/artidoro/qlora/blob/main/qlora.py)

## Differences from original

### airoboros support

Since I am the creator of the various airoboros models, this fork is fairly well catered to the airoboros instruction/response format, and uses the airoboros prompt.

The instructions.jsonl file (or whatever filename you are using), should be a single JSON string per line, newline separated, with "instruction" and "response" values.

Add: `--dataset_format airoboros`

### experimental MPT support

Supports fine-tuning MPT base models via `--mpt True`, but requires a PEFT-compatible base model, e.g.: https://huggingface.co/jondurbin/mpt-30b-qlora-compatible

### epochs instead of steps

I prefer using a fixed number of epochs in training rather than trying to stop are a particular step count.  I removed the `--max_steps` parameter in favor of `--num_train_epochs` (which I usually set to 3)

### experimental flash attention support

Try `flash_qlora.py` instead of `qlora.py`

### lots of stuff removed

MMLU benchmarks, eval, etc.

## Requirements for llama based models

To fine-tune a llama base model, you should use one of these:
- https://huggingface.co/decapoda-research/llama-7b-hf
- https://huggingface.co/decapoda-research/llama-13b-hf
- https://huggingface.co/decapoda-research/llama-30b-hf
- https://huggingface.co/decapoda-research/llama-65b-hf

Then, you __MUST__ replace the `special_tokens_map.json` file and `tokenizer_config.json` file with the ones found in this repo.
