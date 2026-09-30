---
base_model: mlx-community/Qwen2.5-Coder-3B-Instruct-4bit
library_name: mlx
pipeline_tag: text-generation
license: mit
tags:
  - mlx
  - mlx-lm
  - lora
  - text-generation
  - markdown
  - supervised-fine-tuning
  - apple-silicon
datasets:
  - nuprl/MultiPL-E
  - google-research-datasets/mbpp
---

# Qwen2.5-Coder-3B Markdown Reliability LoRA (MLX)

This repository contains the final MLX-LM LoRA adapter from a two-stage local
supervised fine-tuning experiment for strict Markdown structure compliance.
It is designed for the 4-bit Apple Silicon base model
[`mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-Coder-3B-Instruct-4bit)
at revision `3dd939c621c08e5753d5b89f35a2642cd83b98ca`.

On a frozen 240-prompt external Markdown-contract benchmark, full structural-contract
accuracy improved from **8.33%** for the base model to **32.08%** after broad
Run 1 SFT and **82.50%** after the targeted Run 2 correction.

| Model state | Passed | Full contract pass rate |
| --- | ---: | ---: |
| Base model | 20/240 | 8.33% |
| Run 1 broad SFT | 77/240 | 32.08% |
| Run 2 targeted booster | 198/240 | **82.50%** |

## What the 82.50% measures

Each evaluation prompt specified a Markdown contract. A response counted as
correct only if every requested structural rule passed. For example, this
instruction requires both a Python code fence and an outer four-backtick
`markdown` wrapper:

> Include exactly one fenced Python example. Wrap the entire Markdown document
> in one outer four-backtick fence labeled `markdown`.

A passing response looks like this:

~~~~~~text
````markdown
# Example

```python
print("hello")
```
````
~~~~~~

Returning the same inner document without the outer four-backtick wrapper is a
failure. The local evaluator also checks balanced fences, language labels, raw
Markdown source, tables, cited blockquotes, and numbered lists when the prompt
requires them. This is structural-contract accuracy, not a judgment of whether
the programming example is semantically correct.

The full experiment, source code, evaluator, generated-data pipeline, configs,
and report are available at
[`karyboy/mlx-markdown-reliability-lora`](https://github.com/karyboy/mlx-markdown-reliability-lora).

## What is included

- `adapters.safetensors`: final iteration-300 Run 2 adapter (about 114 MiB);
- `adapter_config.json`: complete portable MLX-LM adapter/training metadata;
- `configs/`: frozen experiment, Run 1, and Run 2 configurations;
- `results/`: final external metrics, comparison, and training summary.

The Run 2 adapter is **standalone relative to the base model**. Although Run 2
continued training from Run 1, its saved tensor file contains the complete
final LoRA state; users do not need to download or stack the Run 1 adapter.
Paths such as `data`, `config`, and `resume_adapter_file` in
`adapter_config.json` document the original reproducible training run. The
MLX-LM inference loader uses the model-shape and LoRA fields and does not need
those training files.

## Load with MLX-LM

MLX-LM expects an adapter directory on disk, so download this repository first:

```python
from huggingface_hub import snapshot_download
from mlx_lm import generate, load

BASE_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
ADAPTER_REPO = "logicless/qwen25-coder-3b-markdown-reliability-lora-mlx"

adapter_dir = snapshot_download(
    repo_id=ADAPTER_REPO,
    allow_patterns=["adapter_config.json", "adapters.safetensors"],
)
model, tokenizer = load(BASE_MODEL, adapter_path=adapter_dir)

messages = [{
    "role": "user",
    "content": (
        "Write a concise Markdown guide with a title, a numbered list, "
        "and one fenced Python example. Return the document directly."
    ),
}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=800,
    verbose=False,
)
print(response)
```

Install the tested runtime with:

```zsh
python -m pip install "mlx-lm[train]==0.31.3" "huggingface_hub==1.31.0"
```

## Training summary

The model was trained using assistant-only token loss with LoRA modules on all
36 transformer blocks.

| Setting | Run 1 | Run 2 |
| --- | ---: | ---: |
| Examples | 5,000 broad SFT | 600 targeted booster |
| LoRA rank / scale / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| Learning rate | `1e-5` | `2e-6` |
| Effective batch size | 8 | 8 |
| Selected iterations | 500 | 300 |
| Optimizer updates | 125 | 75 |
| Trained tokens | 261,123 | 91,031 |

Run 1 improved broad structural behavior but failed all 80 external prompts
that required an outer four-backtick `markdown` wrapper. Run 2 used short
targets, exact instruction wording, A2 oversampling, and matched A1/A2 contrast
pairs. It brought external A2 full-contract accuracy from 0/80 to 63/80 while
preserving the A1 and A3 controls.

## Evaluation

The same frozen prompts and deterministic generation settings were used for
the base model and both adapters. Run 2 results by wrapper axis were A1 82.50%,
A2 78.75%, and A3 86.25%. By content-complexity axis they were B1 95%, B2 90%,
B3 80%, and B4 65%.

The score is produced by this project's deterministic structural evaluator.
It checks fence balance, wrapper policy, language-specific code examples, raw
Markdown source, tables, cited blockquotes, and numbered lists. It is not the
upstream dataset authors' official evaluator and does not assess semantic
correctness of arbitrary prose or code.

## Evaluation provenance

The frozen evaluation prompts are a balanced subset derived from
[`latentmd-neurips26/LatentMD`](https://huggingface.co/datasets/latentmd-neurips26/LatentMD)
at a pinned revision. They are evaluation-only and were never used as training
examples or targets. The scorer is the project's own transparent structural
evaluator.

## Training data

The SFT examples are deterministically rendered and programmatically verified
from task-family-disjoint MBPP and MultiPL-E content plans. External benchmark
prompts are never used as training targets. See the GitHub repository's data
attribution and generation scripts for exact source revisions, licenses,
splits, and leakage checks.

## Limitations

- This adapter targets Markdown structure, not factuality or code correctness.
- The 240-prompt result is a controlled experiment rather than a comprehensive
  benchmark of Markdown generation.
- The local evaluator is project-specific.
- This adapter is in MLX-LM format for Apple Silicon; other runtimes may require
  conversion.

## License

The adapter and project code are released under MIT. The original base model
and source datasets retain their respective licenses.
