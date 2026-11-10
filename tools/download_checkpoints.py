"""Restore pinned project checkpoints from Hugging Face and verify SHA-256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def verified(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("countyolo", "sam3", "countvid", "all"),
                        default="countyolo", help="Default: small CountYOLO heads only")
    parser.add_argument("--output-dir", type=Path,
                        help="Override destination; place selected files directly in this directory")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "checkpoints/huggingface.json").read_text("utf-8"))
    selected = [item for item in manifest["files"]
                if args.group == "all" or item.get("group", "countyolo") == args.group]
    for item in selected:
        name = item.get("filename", Path(item["path"]).name)
        target = args.output_dir / name if args.output_dir else ROOT / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the upstream license alongside any mirrored checkpoint.
        license_files = ([item["license"]] if item.get("license") else [])
        license_files += item.get("additional_licenses", [])
        for license_file in license_files:
            license_source = hf_hub_download(
                repo_id=manifest["repo_id"], repo_type=manifest["repo_type"],
                revision=manifest["revision"], filename=license_file,
            )
            license_target = target.parent / "licenses" / Path(license_file).name
            license_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(license_source, license_target)
        if verified(target, item):
            print(f"Verified existing: {name}")
            continue
        source = hf_hub_download(
            repo_id=manifest["repo_id"], repo_type=manifest["repo_type"],
            revision=manifest["revision"], filename=name,
        )
        fd, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output, open(source, "rb") as incoming:
                shutil.copyfileobj(incoming, output)
            staged = Path(temporary)
            if not verified(staged, item):
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
            os.replace(staged, target)
            print(f"Downloaded and verified: {name}")
        finally:
            Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
