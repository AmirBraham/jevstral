"""Download Kev's data at pinned revisions. Verify each file against its Kev manifest."""

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from .records import read_jsonl

HF_DATASET = "jaredpalmer/kev-suites"
HF_REVISION = "cc4bac803e73112689ec327ffa481c519cbc7a05"
KEV_COMMIT = "84847f0a883d900f7de5b7a57eaa341ca7f9a6b4"

HF_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/{{path}}"
KEV_URL = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_COMMIT}/evals/{{path}}"

FILES = (
    "v7/decision-v7/train.jsonl",
    "v7/decision-v7/development.jsonl",
    "night2/dates_unknowable.jsonl",
    "documents-v1/train.jsonl",
    "documents-v1/development.jsonl",
    "hard-v1/train.jsonl",
    "hard-v1/development.jsonl",
    "devtools-v1/train.jsonl",
    "devtools-v1/development.jsonl",
    "round3/transfer-r3/calibration.jsonl",
    "v9/transfer-v9/development.jsonl",
)


def _download(url: str) -> bytes | None:
    """Return the content, or None if the server answers 404."""
    try:
        with urllib.request.urlopen(url, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def fetch(path: str, data_dir: Path) -> Path:
    """Download one file, verify it and write it to data_dir/path."""
    suite, name = path.rsplit("/", 1)
    manifest = _download(KEV_URL.format(path=f"{suite}/manifest.json"))
    if manifest is None:
        raise FileNotFoundError(f"{suite}/manifest.json is not in the Kev repository")
    expected = json.loads(manifest)["files"][name]
    content = _download(HF_URL.format(path=path)) or _download(KEV_URL.format(path=path))
    if content is None:
        raise FileNotFoundError(f"{path} is not in {HF_DATASET} and not in the Kev repository")
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError(f"{path}: SHA-256 {digest} is not the manifest value {expected['sha256']}")
    count = sum(1 for line in content.splitlines() if line.strip())
    if count != expected["records"]:
        raise ValueError(f"{path}: {count} records, the manifest gives {expected['records']}")
    target = data_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def fetch_all(data_dir: Path) -> list[Path]:
    return [fetch(path, data_dir) for path in FILES]


def load(path: str, data_dir: Path) -> list[dict]:
    return read_jsonl(data_dir / path)
