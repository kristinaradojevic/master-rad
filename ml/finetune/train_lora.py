"""Local LoRA fine-tuning with MLX, replacing OpenAI's fine-tuning API
(shut down for self-serve use in 2026 — see the README's Fine-tuning
section). Runs entirely on this Mac's GPU via Apple's MLX framework;
requires Apple Silicon and the extra dependency in ml/requirements.txt.

Base model: mlx-community/Qwen2.5-7B-Instruct-4bit — an open-weight,
already MLX-quantized instruct model, so it works with the app's existing
chat-style prompts (backend/app/services/prompt.py) unchanged. Downloaded
automatically from Hugging Face on first run (~4-5 GB).

Trains on ml/data/finetune/train.jsonl / valid.jsonl (built by
prepare_finetune_data.py — already in the exact {"messages": [...]} format
mlx_lm.lora expects, no conversion needed).

Hyperparameters below are a starting point sized for this dataset and this
machine's memory (M-series, 24GB unified memory):
- max_seq_length=4096: both 16384 and 8192 were tried first and both
  crashed on the very first training step — 16384 hit Metal's ~14.3GB
  per-buffer ceiling, 8192 then hit a true GPU out-of-memory during the
  backward pass, so --grad-checkpoint alone isn't enough headroom on this
  machine at those lengths. Rather than let mlx_lm.lora truncate long
  verdicts against this cap (an earlier run showed 87% of examples cut,
  keeping ~40% of the verdict text on average), prepare_finetune_data.py
  now pre-splits long verdicts into several chunked examples that each fit
  within it — see that script's module docstring. If training still
  crashes with an out-of-memory error, drop this further (e.g. to 2048)
  and rerun prepare_finetune_data.py + this script.
- batch_size=1 avoids wasting memory padding very unevenly-sized examples.
- iters=1500 is roughly 3 epochs over the ~490 chunked examples (490*3),
  for the final model used in the thesis comparison — a quick 500-iter
  pass confirmed training runs cleanly on the chunked data first. Expect
  several hours; run overnight or otherwise somewhere the Mac won't sleep.

Usage:
    python ml/finetune/train_lora.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "ml" / "data" / "finetune"
ADAPTER_DIR = DATA_DIR / "adapters"
FUSED_DIR = DATA_DIR / "fused_model"
RUN_RECORD = DATA_DIR / "lora_run.json"

BASE_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

HYPERPARAMS = {
    "fine_tune_type": "lora",
    "num_layers": 16,
    "batch_size": 1,
    "iters": 1500,
    "max_seq_length": 4096,
}


def _mlx_bin(name: str) -> str:
    # Console scripts installed by pip live alongside the interpreter
    # that ran `pip install` — this finds the venv's copy rather than
    # relying on PATH.
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def main():
    train_file = DATA_DIR / "train.jsonl"
    if not train_file.exists():
        sys.exit(f"{train_file} not found — run prepare_finetune_data.py first.")

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        sys.exit(
            "mlx-lm is not installed. Run:\n"
            "  backend/.venv/bin/pip install -r ml/requirements.txt\n"
            "(Apple Silicon only.)"
        )

    lora_cmd = [
        _mlx_bin("mlx_lm.lora"),
        "--model", BASE_MODEL,
        "--train",
        "--data", str(DATA_DIR),
        "--adapter-path", str(ADAPTER_DIR),
        "--fine-tune-type", HYPERPARAMS["fine_tune_type"],
        "--num-layers", str(HYPERPARAMS["num_layers"]),
        "--batch-size", str(HYPERPARAMS["batch_size"]),
        "--iters", str(HYPERPARAMS["iters"]),
        "--max-seq-length", str(HYPERPARAMS["max_seq_length"]),
        "--mask-prompt",
        "--grad-checkpoint",
    ]
    print("Starting LoRA training (this can take a while — progress prints below):")
    print(" ".join(lora_cmd), "\n")
    start = time.time()
    subprocess.run(lora_cmd, check=True)
    train_seconds = round(time.time() - start)

    fuse_cmd = [
        _mlx_bin("mlx_lm.fuse"),
        "--model", BASE_MODEL,
        "--adapter-path", str(ADAPTER_DIR),
        "--save-path", str(FUSED_DIR),
    ]
    print("\nFusing the adapter into a standalone model:")
    print(" ".join(fuse_cmd), "\n")
    subprocess.run(fuse_cmd, check=True)

    RUN_RECORD.write_text(
        json.dumps(
            {"base_model": BASE_MODEL, **HYPERPARAMS, "train_seconds": train_seconds},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDone in {train_seconds}s. Fused model saved to {FUSED_DIR}")
    print("\nServe it with:")
    print(f"  backend/.venv/bin/mlx_lm.server --model {FUSED_DIR} --port 8081 --max-tokens 8000")
    print("\nThen add to backend/.env:")
    print("  LOCAL_MLX_BASE_URL=http://localhost:8081/v1")


if __name__ == "__main__":
    main()
