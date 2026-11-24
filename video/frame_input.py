"""Shared frame selection and reproducibility metadata for UI and terminal."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re


def resolve_input(folder, text=None, max_frames=0):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Không thấy thư mục frame: {folder}")
    if max_frames < 0:
        raise ValueError("--max-frames phải >=0; 0 nghĩa là toàn bộ")
    frames = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    frames.sort(key=lambda p: (tuple(int(x) for x in re.findall(r"\d+", p.stem)), p.name))
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError("Thư mục không có frame JPG/PNG")
    if not text:
        # Read prompt names only. Ground-truth counts never enter inference.
        prompts = set()
        for ancestor in folder.parents:
            annotations = ancestor / "anno"
            if annotations.is_dir():
                for path in annotations.glob("*-count-gt.json"):
                    entry = json.loads(path.read_text(encoding="utf-8-sig")).get(folder.name, {})
                    if isinstance(entry, dict):
                        prompts.update(entry)
                break
        if len(prompts) != 1:
            raise ValueError('Không xác định được một tên vật từ annotation. Thêm --text "tên vật".')
        text = next(iter(prompts))
    text = text.strip()
    if not text:
        raise ValueError("Tên vật không được để trống")
    return folder, frames, text


@lru_cache(maxsize=32)
def _file_hash(path, size, modified_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _file_hash(str(path), stat.st_size, stat.st_mtime_ns)


def run_metadata(frames, text, checkpoint, sam_checkpoint, min_mass, window, batch_size, device, frame_dir=None):
    paths = [Path(f).resolve() for f in frames]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(json.dumps([path.name, file_hash(path)], ensure_ascii=False).encode("utf-8"))
    return dict(text=text, frame_dir=str(Path(frame_dir).resolve()) if frame_dir else None,
                frame_names=[p.name for p in paths], frame_content_sha256=digest.hexdigest(),
                checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=file_hash(checkpoint),
                sam_checkpoint=str(Path(sam_checkpoint).resolve()), sam_checkpoint_sha256=file_hash(sam_checkpoint),
                min_mass=float(min_mass), window=int(window), batch_size=int(batch_size), device=str(device))
