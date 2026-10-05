"""Modal entry points. All data, model and GPU work runs here, not on the local computer.

modal run modal_app.py::prepare_data
modal run modal_app.py::inspect_data
modal run modal_app.py::train --stage 1 --smoke
modal deploy modal_app.py          (after each code change)
python modal_app.py train 1        (full stage; returns at once, the job runs on Modal)
python modal_app.py calibrate
modal run modal_app.py::check_hub
modal run modal_app.py::publish --stage 1
modal run modal_app.py::publish_model_card
modal run modal_app.py::track_finished_stage --stage 1
modal run modal_app.py::evaluate_stage --stage 2
modal run modal_app.py::check_inference
modal run modal_app.py::check_cache
python modal_app.py build_suite      (CPU, hours; needs access to cais/hle)
modal run modal_app.py::bench_sample --n 100
python modal_app.py bench_full       (only after the sample, with an approved cost)
"""

import contextlib
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("jevstral")
HARNESS_COMMIT = "87d4650b42b377c0291a89c1f1a879f9b31082bf"  # the same commit as the bench group in pyproject.toml

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # uv fetches the pinned Decision Index harness from GitHub
    .uv_sync(groups=["train", "bench"], extras=["inference"])
    # The harness reads its hub/ files (manifest, exclusions) from <site-packages>/hub, a path that only exists in a
    # git checkout. Copy them from the same pinned commit.
    .run_commands(
        f"git clone --quiet https://github.com/apolinario/decision-index.git /tmp/decision-index"
        f" && git -C /tmp/decision-index checkout --quiet {HARNESS_COMMIT}"
        " && cp -r /tmp/decision-index/hub /.uv/.venv/lib/python3.12/site-packages/hub"
        " && rm -rf /tmp/decision-index"
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("jevstral")
)

hf_cache = modal.Volume.from_name("jevstral-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("jevstral-data", create_if_missing=True)
runs = modal.Volume.from_name("jevstral-runs", create_if_missing=True)
bench = modal.Volume.from_name("jevstral-bench", create_if_missing=True)

DATA_DIR = Path("/data")
RUNS_DIR = Path("/runs")
VOLUMES = {"/cache": hf_cache, str(DATA_DIR): data, str(RUNS_DIR): runs}
BENCH_DIR = Path("/bench")
SUITE_DIR = BENCH_DIR / "suite-0.2"
FINAL = RUNS_DIR / "main" / "stage4" / "final"
# The Decision Index maintainers measure latency on one RTX PRO 6000. The same GPU makes our latency comparable.
BENCH_GPU = "RTX-PRO-6000"
SECRETS = [modal.Secret.from_name("huggingface")]
WANDB = modal.Secret.from_name("wandb")
HOUR = 3600


def persist() -> None:
    hf_cache.commit()
    runs.commit()


@app.function(image=image, volumes={str(DATA_DIR): data}, timeout=HOUR)
def prepare_data() -> None:
    from jevstral import suites

    for path in suites.fetch_all(DATA_DIR):
        print("verified", path)
    data.commit()


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, memory=32768, timeout=HOUR)
def inspect_data() -> None:
    """CPU only. Measure the delimiter embeddings and encode all stages without a model."""
    from jevstral.encode import Encoder
    from jevstral.model import delimiter_report, load_tokenizer
    from jevstral.stages import STAGES
    from jevstral.train import encoding_report

    print(delimiter_report(Encoder(load_tokenizer()).delimiter_ids))
    hf_cache.commit()
    print(encoding_report(list(STAGES.values()), DATA_DIR))


@app.function(
    image=image,
    gpu="H100",
    volumes=VOLUMES,
    secrets=[*SECRETS, WANDB],
    memory=65536,
    # No `retries`: Modal reruns preempted jobs itself, and a retry after an exception in our code fails again.
    timeout=8 * HOUR,
)
def train_stage(stage: int, smoke: bool, git_commit: str) -> None:
    from jevstral.stages import STAGES
    from jevstral.train import run_stage

    run_stage(STAGES[stage], DATA_DIR, RUNS_DIR, git_commit, smoke=smoke, persist=persist)


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, memory=65536, timeout=2 * HOUR)
def evaluate_stage(stage: int) -> None:
    """Measure a finished stage on all four development files, without training."""
    import json

    from jevstral.stages import STAGES
    from jevstral.train import evaluate_checkpoint

    metrics = evaluate_checkpoint(stage, STAGES[4].dev_files, DATA_DIR, RUNS_DIR)
    runs.commit()
    print(json.dumps(metrics, indent=2))


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, memory=65536, timeout=HOUR)
def check_inference() -> None:
    """Latency of the fp32 and bf16-merged paths, and the difference between their answers."""
    import json

    from jevstral.inference import inference_report

    report = inference_report(RUNS_DIR / "main" / "stage4" / "final", DATA_DIR)
    runs.commit()
    print(json.dumps(report, indent=2))


@app.function(
    image=image, gpu=BENCH_GPU, volumes={**VOLUMES, str(BENCH_DIR): bench}, secrets=SECRETS, memory=65536, timeout=HOUR
)
def check_cache(n: int = 100) -> None:
    """State cache against the plain path, on the Decision Index sample of n requests and on development records."""
    import gzip
    import json

    from jevstral.inference import cache_report

    with gzip.open(BENCH_DIR / f"sample-{n}.jsonl.gz", "rt") as file:
        requests = [json.loads(line) for line in file if line.strip()]
    report = cache_report(FINAL, DATA_DIR, requests)
    runs.commit()
    print(json.dumps(report, indent=2))


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, memory=65536, timeout=2 * HOUR)
def calibrate_stage4() -> None:
    from jevstral.calibrate import run_calibration

    run_calibration(DATA_DIR, RUNS_DIR, persist=persist)


@app.function(image=image, secrets=SECRETS, timeout=600)
def check_hub() -> None:
    """Verify that the Hugging Face token can write: create the private repository if it does not exist."""
    from jevstral.publish import ensure_repo, token_report

    print(token_report())
    print("repository ready:", ensure_repo())


@app.function(image=image, secrets=SECRETS, timeout=600)
def make_model_public() -> None:
    from jevstral.publish import make_public

    print("public:", make_public())


@app.function(image=image, secrets=SECRETS, timeout=600)
def upload_card(card: str) -> None:
    from jevstral.publish import publish_card

    print("model card:", publish_card(card))


@app.function(image=image, volumes={str(RUNS_DIR): runs}, secrets=SECRETS, timeout=HOUR)
def upload_stage(stage: int) -> None:
    from jevstral.publish import publish_stage

    print("uploaded:", publish_stage(stage, RUNS_DIR))


@app.function(image=image, volumes={str(RUNS_DIR): runs}, secrets=[WANDB], timeout=600)
def track_finished_stage(stage: int) -> None:
    """Send the log and metrics of a stage that trained without W&B (stage 1) to a W&B run."""
    from jevstral.tracking import upload_finished_stage

    print("run:", upload_finished_stage(RUNS_DIR / "main" / f"stage{stage}"))


@contextlib.contextmanager
def committing(volume: modal.Volume, every_seconds: int):
    """Commit `volume` every `every_seconds` and at the end, so a stopped job keeps its files."""
    import threading

    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(every_seconds):
            volume.commit()

    threading.Thread(target=loop, daemon=True).start()
    try:
        yield
    finally:
        stop.set()
        volume.commit()


@contextlib.contextmanager
def scoring_partial(out: str, every_seconds: int):
    """Score the results so far every `every_seconds`, into <out>/partial/. A stopped run keeps its last partial
    scores. A scoring error is printed and does not stop the run."""
    import threading

    stop = threading.Event()
    results = BENCH_DIR / out / "results.jsonl"
    partial = BENCH_DIR / out / "partial"

    def score() -> None:
        if not results.exists():
            return
        text = results.read_text(encoding="utf-8")
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "results.jsonl").write_text(text[: text.rfind("\n") + 1], encoding="utf-8")  # complete lines only
        try:
            harness(
                "score",
                "--results",
                str(partial / "results.jsonl"),
                "--suite-dir",
                str(SUITE_DIR),
                "--engine",
                "jevstral",
                "--out",
                str(partial),
            )
        except subprocess.CalledProcessError as error:
            print(f"partial scoring failed: {error}", flush=True)
        bench.commit()

    def loop() -> None:
        while not stop.wait(every_seconds):
            score()

    threading.Thread(target=loop, daemon=True).start()
    try:
        yield
    finally:
        stop.set()


def harness(*args: str) -> None:
    """Run one command of the Decision Index harness and stop on failure."""
    import sys

    command = [sys.executable, "-m", "decision_index", *args]
    print("$", " ".join(command), flush=True)
    # /root holds the jevstral package in the container; the harness imports jevstral.engine from there.
    env = {
        **os.environ,
        "HF_HUB_DISABLE_XET": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, ["/root", os.environ.get("PYTHONPATH")])),
    }
    subprocess.run(command, check=True, cwd=BENCH_DIR, env=env)


@app.function(
    image=image,
    volumes={str(BENCH_DIR): bench},
    secrets=SECRETS,
    memory=32768,
    timeout=8 * HOUR,
    nonpreemptible=True,  # a preemption on 2026-10-04 lost 30 minutes of downloads
)
def build_suite(reuse_work: bool = False) -> None:
    """Rebuild the Decision Index 0.2 suite from its public sources and verify it (CPU, about 7 GB of downloads).

    The Hugging Face token must have access to the gated dataset cais/hle.
    """
    built = "work/artifacts/benchmark-suite/release-v2-rebuilt"
    # Downloads already skip files that exist with the right SHA-256, so only the normalization is skipped.
    reuse = ["--skip-normalize"] if reuse_work else []
    with committing(bench, every_seconds=300):
        harness("suite", "rebuild", "--work", "work", *reuse)
    harness(
        "suite",
        "import",
        "--dir",
        str(SUITE_DIR),
        "--rows",
        f"{built}/selected-rows.jsonl.gz",
        "--added-rows",
        f"{built}/added-rows.jsonl.gz",
    )
    bench.commit()


@app.function(
    image=image,
    gpu=BENCH_GPU,
    volumes={**VOLUMES, str(BENCH_DIR): bench},
    secrets=SECRETS,
    memory=65536,
    timeout=2 * HOUR,
)
def bench_sample(n: int = 100, path: str = "bf16-merged") -> None:
    """Run Jevstral on a sample of n requests of the suite and score it. Use the output to estimate the full run."""
    sample = f"sample-{n}.jsonl.gz"
    out = f"runs/sample-{n}-{path}"
    harness("suite", "sample", "--dir", str(SUITE_DIR), "--n", str(n), "--out", sample)
    harness(
        "run",
        "--engine",
        "jevstral.engine:JevstralEngine",
        "--option",
        f"checkpoint={FINAL}",
        "--option",
        f"path={path}",
        "--suite-dir",
        str(SUITE_DIR),
        "--rows",
        sample,
        "--out",
        out,
        "--fresh",
    )
    harness(
        "score",
        "--results",
        f"{out}/results.jsonl",
        "--suite-dir",
        str(SUITE_DIR),
        "--engine",
        "jevstral",
        "--out",
        out,
    )
    bench.commit()


@app.function(
    image=image,
    gpu=BENCH_GPU,
    volumes={**VOLUMES, str(BENCH_DIR): bench},
    secrets=SECRETS,
    memory=65536,
    timeout=24 * HOUR,
)
def bench_full(path: str = "bf16-merged") -> None:
    """Run and score the full suite. Resumes from an existing results.jsonl. Start it with start_job (deployed app)."""
    out = f"runs/full-{path}"
    with committing(bench, every_seconds=600), scoring_partial(out, every_seconds=1800):
        harness(
            "pipeline",
            "--engine",
            "jevstral.engine:JevstralEngine",
            "--option",
            f"checkpoint={FINAL}",
            "--option",
            f"path={path}",
            "--suite-dir",
            str(SUITE_DIR),
            "--out",
            out,
            "--compact",
        )


def local_git_commit() -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    return commit + ("-dirty" if dirty else "")


@app.local_entrypoint()
def train(stage: int, smoke: bool = False) -> None:
    """Smoke runs only. Full stages start from the deployed app (see start_job below): an ephemeral app depends on
    the local network, and a lost connection cancelled stage 4 on 2026-10-04."""
    if not smoke:
        raise SystemExit("Start full stages with: modal deploy modal_app.py && python modal_app.py train <stage>")
    train_stage.remote(stage, smoke, local_git_commit())


@app.local_entrypoint()
def publish(stage: int) -> None:
    upload_stage.remote(stage)


@app.local_entrypoint()
def publish_model_card() -> None:
    upload_card.remote(Path("docs/model-card.md").read_text())


def start_job(argv: list[str]) -> None:
    """Start a long job on the deployed app and return at once. The job does not depend on this process.

    python modal_app.py train <stage>
    python modal_app.py calibrate
    """
    if argv[:1] == ["train"] and len(argv) == 2:
        call = modal.Function.from_name("jevstral", "train_stage").spawn(int(argv[1]), False, local_git_commit())
    elif argv == ["calibrate"]:
        call = modal.Function.from_name("jevstral", "calibrate_stage4").spawn()
    elif argv[:1] == ["build_suite"]:
        call = modal.Function.from_name("jevstral", "build_suite").spawn(reuse_work="--reuse" in argv)
    elif argv == ["bench_full"]:
        call = modal.Function.from_name("jevstral", "bench_full").spawn()
    else:
        raise SystemExit("usage: python modal_app.py train <stage> | calibrate | build_suite [--reuse] | bench_full")
    print(f"started {call.object_id}. Follow it at https://modal.com/apps (app jevstral, deployed).")


if __name__ == "__main__":
    import sys

    start_job(sys.argv[1:])
