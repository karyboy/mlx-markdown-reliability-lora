# Publishing guide

This project is published as two linked public repositories:

- GitHub source and experiment report:
  `karyboy/mlx-markdown-reliability-lora`
- Hugging Face final adapter:
  `logicless/qwen25-coder-3b-markdown-reliability-lora-mlx`

The GitHub repository intentionally excludes `data/`, `outputs/`, `adapters/`,
the Python environment, and downloaded base-model weights. The Hugging Face
upload script sends only the selected final Run 2 adapter and its supporting
metadata—not the six intermediate 114 MB checkpoints.

## 1. Run the publication audit

```zsh
cd /path/to/markdown-reliability
../.venv/bin/python scripts/audit_publication.py
```

The command fails if a publication file contains a likely credential, private
machine path, oversized file, or other blocked artifact.

## 2. Publish GitHub

If GitHub CLI is authenticated:

```zsh
cd /path/to/markdown-reliability
git init
git add .gitignore README.md REPORT.md PUBLISHING.md LICENSE CITATION.cff \
  THIRD_PARTY_DATA.md requirements.txt assets configs results scripts src tests \
  publish/huggingface/README.md
git commit -m "Publish MLX Markdown reliability LoRA experiment"
git branch -M main
gh repo create karyboy/mlx-markdown-reliability-lora \
  --public --source=. --remote=origin --push
```

If the empty GitHub repository was created in the browser instead, replace the
last `gh` command with:

```zsh
git remote add origin https://github.com/karyboy/mlx-markdown-reliability-lora.git
git push -u origin main
```

Do not use `git add .`; the explicit list is an additional safeguard against
publishing generated or private files.

## 3. Publish the Hugging Face adapter

Authenticate once if needed without putting a token in a command or file:

```zsh
hf auth login
```

Then run:

```zsh
cd /path/to/markdown-reliability
../.venv/bin/python scripts/publish_huggingface.py \
  --repo-id logicless/qwen25-coder-3b-markdown-reliability-lora-mlx
```

The script creates a public model repository, validates the adapter SHA-256,
and uploads the following allowlisted files:

- `adapters.safetensors` and `adapter_config.json`;
- the Hugging Face model card and MIT license;
- Run 1 and Run 2 training configs;
- final external metrics and training summary.

It reads the cached Hugging Face login or `HF_TOKEN`; it never records or
prints the token.

## 4. Verify the links

- <https://github.com/karyboy/mlx-markdown-reliability-lora>
- <https://huggingface.co/logicless/qwen25-coder-3b-markdown-reliability-lora-mlx>

Both cards link to the other repository so readers can move between the full
experiment and the downloadable adapter.
