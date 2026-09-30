# Third-party data attribution

The final Markdown SFT corpus uses third-party programming tasks only as
structured content plans. The Markdown prompts, formatting contracts, rendered
answers, corruptions, split manifest, and structural evaluator are original to
this project.

## MultiPL-E

- Dataset: <https://huggingface.co/datasets/nuprl/MultiPL-E>
- Revision: `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`
- License: MIT
- Use: language-specific starter-code prompts, test harnesses, and task-family
  identifiers from eight MBPP-derived language configurations.

## Mostly Basic Python Problems (MBPP)

- Dataset: <https://huggingface.co/datasets/google-research-datasets/mbpp>
- Revision: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`
- Configuration: `full`
- License: CC BY 4.0
- Use: original task descriptions, Python reference implementations, and test
  assertions.

Training and validation are split by the underlying MBPP task ID before any
Markdown examples are rendered. Translations of one task family cannot cross
the split boundary.
