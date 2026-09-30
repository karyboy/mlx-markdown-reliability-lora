# Experiment report: improving Markdown contract reliability with local LoRA SFT

## Executive summary

This project asks a narrow post-training question: can a small, locally trained
language model become substantially more reliable at producing Markdown under
strict structural constraints?

The experiment starts from
[`mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-Coder-3B-Instruct-4bit),
a 3.09-billion-parameter 4-bit model, and trains LoRA adapters locally with
MLX-LM on an Apple Silicon Mac. Evaluation uses the same frozen 240-prompt
external Markdown-contract benchmark, prompt construction, deterministic
generation settings, and local structural evaluator for every model version.

The final result is a change in full structural-contract pass rate from
**8.33% to 82.50%**:

| Model state | Passed | Pass rate | Change from prior state |
| --- | ---: | ---: | ---: |
| Base model | 20/240 | 8.33% | — |
| Run 1: broad Markdown SFT | 77/240 | 32.08% | +23.75 points |
| Run 2: targeted wrapper booster | 198/240 | **82.50%** | +50.42 points |

![Full structural contract pass rate](assets/experiment-results.svg)

This is a format-reliability result, not a general measure of answer quality.
The evaluator verifies requested Markdown structure; it does not determine
whether arbitrary prose or programming content is semantically correct.

## 1. Objective

Markdown can look correct to a reader while violating a machine-consumable
contract. Nested fences, visible raw Markdown examples, exact wrapper rules,
tables, numbered lists, and cited blockquotes create interactions that are easy
for a model to mishandle.

The objective was therefore not merely to produce readable Markdown. It was to
increase the probability that one generated response satisfies **all** of the
structural constraints stated in its prompt.

The experiment used a 3 x 4 contract grid:

| Axis | Contract |
| --- | --- |
| A1 | Return the document directly and do not use an outer wrapper. |
| A2 | Wrap the entire document in a four-backtick `markdown` fence. |
| A3 | No explicit outer-wrapper requirement. |
| B1 | Exactly one language-specific fenced code example. |
| B2 | A code example plus visible raw Markdown fence source. |
| B3 | Multiple code examples plus multiple raw-source blocks. |
| B4 | Multiple code examples, a table, a cited blockquote, and a numbered list. |

The primary metric is the proportion of responses passing their full A x B
contract. Individual checks are also reported for diagnosis.

## 2. Reproducible evaluation contract

The following elements were frozen before comparing model versions:

- base model ID and model revision;
- 240 evaluation prompts, balanced at 20 prompts per A x B cell;
- chat template and user prompt text;
- maximum output length, temperature, top-p, and random seed;
- deterministic structural evaluator and scoring rules.

For provenance, the external prompts are a balanced subset derived from the
[`latentmd-neurips26/LatentMD`](https://huggingface.co/datasets/latentmd-neurips26/LatentMD)
repository at revision `30fa204394b7214501ddc280a198c26fccf166df`.
They are used only for evaluation. No external prompt or model response is used
as an SFT target.

The project evaluator is intentionally transparent and local. It checks:

- parse success and balanced fences;
- compliance with the outer-wrapper policy;
- required language-specific code examples;
- visible raw Markdown source;
- tables, cited blockquotes, and numbered lists when requested.

It is not the upstream dataset authors' official evaluator and it does not
execute or semantically grade programming solutions.

## 3. Training data construction

The broad Run 1 corpus contains 5,000 training examples and 500 internal
validation examples. Rather than treating a small set of handwritten targets
as many unrelated examples, the project builds targets deterministically from
200 underlying programming-task families:

- 160 task families are assigned to training;
- 40 disjoint task families are assigned to validation;
- no task family or target hash crosses the split;
- every rendered target must pass the same structural verifier used by the
  project before it is accepted.

The content plans use MBPP tasks and eight MBPP-derived MultiPL-E language
configurations: C++, Java, JavaScript, Go, Rust, Ruby, Shell, and TypeScript.
The output mix is 60% direct generation, 30% malformed-Markdown repair, and
10% JSON-to-Markdown transformation. Exact source revisions and licensing are
recorded in [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md).

## 4. Baseline

The unmodified base model passed **20 of 240 prompts (8.33%)**. Several
individual elements were often present, but composing all requested elements
correctly was rare.

| Check | Base pass rate |
| --- | ---: |
| Balanced fences | 92.08% |
| Language code examples | 26.67% |
| Raw Markdown source | 65.42% |
| Cited blockquote | 77.50% |
| Outer-wrapper policy | 38.33% |

This distinction matters: high pass rates on isolated checks do not imply that
the full multi-part contract is satisfied.

## 5. Run 1: broad LoRA SFT

Run 1 applied assistant-only-loss LoRA SFT across all 36 transformer blocks.
The selected checkpoint used:

| Setting | Value |
| --- | ---: |
| LoRA rank / scale / dropout | 16 / 32 / 0.05 |
| Learning rate | `1e-5` |
| Micro-batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Selected iteration | 500 |
| Optimizer updates | 125 |
| Trained tokens | 261,123 |
| Trainable parameters | 29.934M (0.97%) |

The original ceiling was 2,500 micro-batches. Training was stopped at iteration
510 and the complete iteration-500 checkpoint was selected because sampled
training and validation loss had reached approximately zero.

Run 1 improved the full-contract result to **77/240 (32.08%)**. It also raised
language-code compliance from 26.67% to 64.17% and raw-source compliance from
65.42% to 79.17%.

However, failure analysis exposed a systematic error: **all 80 A2 examples
failed**. The adapter generated the inner document directly instead of opening
the required outer four-backtick `markdown` wrapper.

### Why near-zero loss was not enough

Teacher-forced token loss averages prediction error over a sequence. The A2
decision is concentrated in the first few output tokens. A model can predict
almost every inner-document token correctly and still fail the entire contract
by making one wrong opening decision. Consequently, near-zero validation loss
on a synthetic distribution did not guarantee correct free-running behavior on
the external A2 prompts.

This was the central diagnostic insight of the experiment: sequence-level
evaluation caught a failure mode that token-level loss concealed.

## 6. Run 2: targeted wrapper correction

Run 2 continued from the Run 1 adapter with a deliberately small 600-example
booster and a 120-example disjoint internal validation set.

The booster changed the learning signal in five ways:

1. A2 was oversampled: 360 A2, 120 A1, and 120 A3 training examples.
2. It included 120 matched A1/A2 pairs with identical content, so only the
   wrapper instruction and correct boundary tokens changed.
3. Eighty percent of training examples were direct generation using the exact
   external instruction wording.
4. Targets were shorter, increasing the contribution of opening and closing
   wrapper tokens to average token loss.
5. The learning rate was reduced from `1e-5` to `2e-6` to make a focused
   correction without erasing the broader Run 1 behavior.

Run 2 used 300 micro-batches, 75 optimizer updates, and 91,031 trained tokens.
Its held-out loss decreased from 1.334 at the first evaluation to 0.266 at
iteration 300. On the booster’s internal generation-only validation set, full
contract accuracy rose from **29/120 (24.17%)** with Run 1 to **116/120
(96.67%)** with Run 2; A2 improved from **0/72 to 70/72**.

## 7. Final external results

On the unchanged 240-prompt external set, Run 2 passed **198/240 (82.50%)**.
This is a **74.17 percentage-point** absolute improvement and a **9.9x**
relative improvement over the base model.

![Run 2 contract breakdown](assets/run2-breakdown.svg)

| Outer-wrapper axis | Passed | Pass rate |
| --- | ---: | ---: |
| A1: wrapper prohibited | 66/80 | 82.50% |
| A2: wrapper required | 63/80 | 78.75% |
| A3: wrapper unspecified | 69/80 | 86.25% |

| Content-complexity axis | Passed | Pass rate |
| --- | ---: | ---: |
| B1 | 57/60 | 95.00% |
| B2 | 54/60 | 90.00% |
| B3 | 48/60 | 80.00% |
| B4 | 39/60 | 65.00% |

Every response satisfied its A1/A2/A3 outer-wrapper policy. The remaining
full-contract failures are concentrated in the harder B3 and B4 compositions,
especially language-specific example counts and visible raw Markdown source.

## 8. What changed between Run 1 and Run 2

Run 2 was not simply “more training.” It was a targeted intervention based on
an observed failure slice.

| Design choice | Run 1 | Run 2 |
| --- | --- | --- |
| Purpose | Broad Markdown reliability | Repair wrapper decision |
| Training examples | 5,000 | 600 |
| A-axis balance | Approximately balanced | A2 oversampled 3:1 vs each control |
| Matched A1/A2 contrasts | Incidental | 120 explicit pairs |
| Sequence length | Up to 4,096 tokens | Up to 1,280 tokens |
| Learning rate | `1e-5` | `2e-6` |
| Starting point | Base model | Run 1 adapter |
| External full-contract pass rate | 32.08% | **82.50%** |

The result supports an iterative post-training workflow: train broadly, measure
free-running behavior, slice errors by contract, and construct a narrow second
curriculum that directly contrasts the failed decision with nearby controls.

## 9. Limitations

- The evaluation contains 240 prompts, enough for a controlled portfolio
  experiment but not a comprehensive benchmark.
- The local structural evaluator is deterministic but project-specific and is
  not the upstream dataset's official evaluator.
- Generated training targets are structurally verified; their programming
  prose is not a measure of solution correctness.
- The model and adapter are optimized for MLX-LM on Apple Silicon. Other
  runtimes require conversion or a compatible adapter loader.
- Run 2 specifically targets a diagnosed wrapper failure. Additional held-out
  prompt styles would be needed to establish broader generalization.

## 10. Reproducibility and artifacts

- Source, configs, evaluator, and exact metrics:
  [GitHub repository](https://github.com/karyboy/mlx-markdown-reliability-lora)
- Final standalone Run 2 adapter and model card:
  [Hugging Face model repository](https://huggingface.co/logicless/qwen25-coder-3b-markdown-reliability-lora-mlx)
- Frozen experiment settings: [`configs/experiment.yaml`](configs/experiment.yaml)
- Run 1 config: [`configs/lora.yaml`](configs/lora.yaml)
- Run 2 config: [`configs/lora_wrapper_booster.yaml`](configs/lora_wrapper_booster.yaml)
- Machine-readable comparison: [`results/final_comparison.json`](results/final_comparison.json)

The GitHub repository deliberately excludes downloaded datasets, generated
model responses, base-model weights, and adapter checkpoints. The Hugging Face
repository contains only the final Run 2 adapter plus the metadata required to
load and understand it.
