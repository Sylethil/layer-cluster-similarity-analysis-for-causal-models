"""
Run this once to download all models into D:/252 Project/models/.
Safe to re-run — skips any model whose directory already exists.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

# ── Config ────────────────────────────────────────────────────────────────────

TOKEN_FILE = Path(__file__).parent / "hf_token.txt"
MODELS_DIR = Path(__file__).parent / "models"

MODELS = [
    {
        "repo_id": "gpt2-large",
        "local_dir": MODELS_DIR / "gpt2-large-hf",
    },
    {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": MODELS_DIR / "llama-3.1-8b-instruct",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_token() -> str:
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(f"HF token file not found: {TOKEN_FILE}")
    lines = [l.strip() for l in TOKEN_FILE.read_text().splitlines() if l.strip().startswith("hf_")]
    if not lines:
        raise ValueError(f"No HuggingFace token (starting with 'hf_') found in {TOKEN_FILE}")
    return lines[0]


def already_downloaded(local_dir: Path) -> bool:
    """Consider a model downloaded if its directory has at least one weight file."""
    if not local_dir.exists():
        return False
    weight_extensions = {".safetensors", ".bin", ".pt", ".gguf"}
    return any(f.suffix in weight_extensions for f in local_dir.rglob("*"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    token = load_token()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for entry in MODELS:
        repo_id: str = entry["repo_id"]
        local_dir: Path = entry["local_dir"]

        if already_downloaded(local_dir):
            print(f"[skip] {repo_id} — already in {local_dir}")
            continue

        print(f"\n[download] {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            tqdm_class=None,   # uses huggingface_hub's built-in tqdm bars
        )
        print(f"[done] {repo_id}")

    print("\nAll models ready.")


if __name__ == "__main__":
    main()
