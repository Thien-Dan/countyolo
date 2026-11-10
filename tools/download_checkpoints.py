"""Restore pinned CountYOLO heads from Hugging Face and verify SHA-256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "checkpoints/huggingface.json").read_text("utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        name = Path(item["path"]).name
        target = args.output_dir / name
        if (target.is_file() and target.stat().st_size == item["size"]
                and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]):
            print(f"Verified existing: {name}")
            continue
        source = hf_hub_download(
            repo_id=manifest["repo_id"], repo_type=manifest["repo_type"],
            revision=manifest["revision"], filename=name,
        )
        fd, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=args.output_dir)
        try:
            with os.fdopen(fd, "wb") as output, open(source, "rb") as incoming:
                shutil.copyfileobj(incoming, output)
            staged = Path(temporary)
            if (staged.stat().st_size != item["size"]
                    or hashlib.sha256(staged.read_bytes()).hexdigest() != item["sha256"]):
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
            os.replace(staged, target)
            print(f"Downloaded and verified: {name}")
        finally:
            Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
