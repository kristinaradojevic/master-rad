"""Upload the prepared JSONL dataset to OpenAI and run a fine-tuning job.

Usage:
    python ml/preprocess/run_finetune_job.py            # start a new job, then poll until done
    python ml/preprocess/run_finetune_job.py --status   # just check the last job started here

The resulting model id is saved to ml/data/finetune/job.json. Copy it into
FINETUNED_MODEL_ID in backend/.env to make it selectable as a model in the
app (see backend/app/services/llm/registry.py).
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / "backend" / ".env")

OUT_DIR = REPO_ROOT / "ml" / "data" / "finetune"
TRAIN_FILE = OUT_DIR / "train.jsonl"
VALID_FILE = OUT_DIR / "valid.jsonl"
JOB_RECORD = OUT_DIR / "job.json"

BASE_MODEL = "gpt-4o-mini-2024-07-18"
SUFFIX = "master-rad-verdicts"
POLL_SECONDS = 30


def _print_status(job) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] status={job.status}")


def check_status(client: OpenAI):
    if not JOB_RECORD.exists():
        sys.exit(f"No job recorded yet at {JOB_RECORD} — run without --status first.")
    record = json.loads(JOB_RECORD.read_text(encoding="utf-8"))
    job = client.fine_tuning.jobs.retrieve(record["job_id"])
    _print_status(job)
    if job.status == "succeeded":
        print(f"Fine-tuned model: {job.fine_tuned_model}")
    return job


def start_job(client: OpenAI):
    if not TRAIN_FILE.exists():
        sys.exit(f"{TRAIN_FILE} not found — run prepare_finetune_data.py first.")

    train_upload = client.files.create(file=TRAIN_FILE.open("rb"), purpose="fine-tune")
    print(f"Uploaded training file: {train_upload.id}")

    kwargs = {"training_file": train_upload.id, "model": BASE_MODEL, "suffix": SUFFIX}
    if VALID_FILE.exists() and VALID_FILE.stat().st_size > 0:
        valid_upload = client.files.create(file=VALID_FILE.open("rb"), purpose="fine-tune")
        print(f"Uploaded validation file: {valid_upload.id}")
        kwargs["validation_file"] = valid_upload.id

    job = client.fine_tuning.jobs.create(**kwargs)
    print(f"Started fine-tuning job: {job.id}")

    JOB_RECORD.write_text(
        json.dumps({"job_id": job.id, "base_model": BASE_MODEL}, indent=2), encoding="utf-8"
    )
    return job


def poll_until_done(client: OpenAI, job):
    while job.status not in ("succeeded", "failed", "cancelled"):
        time.sleep(POLL_SECONDS)
        job = client.fine_tuning.jobs.retrieve(job.id)
        _print_status(job)

    record = json.loads(JOB_RECORD.read_text(encoding="utf-8"))
    record["status"] = job.status
    record["fine_tuned_model"] = job.fine_tuned_model
    JOB_RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if job.status == "succeeded":
        print(f"\nDone. Fine-tuned model: {job.fine_tuned_model}")
        print("Set FINETUNED_MODEL_ID to this value in backend/.env to use it in the app.")
    else:
        print(f"\nJob ended with status={job.status} — check the OpenAI dashboard for details.")


def main():
    client = OpenAI()  # reads OPENAI_API_KEY from the environment

    if "--status" in sys.argv:
        check_status(client)
        return

    job = start_job(client)
    poll_until_done(client, job)


if __name__ == "__main__":
    main()
