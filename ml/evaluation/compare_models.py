"""Model comparison for the thesis.

Runs the same set of verdict inputs through every configured model
(via the backend's provider registry) and saves the outputs side by side,
so they can be evaluated — manually by legal experts, and/or automatically
(structure completeness, style similarity to real verdicts, etc.).

Usage:
    # from the backend/ directory so the app package is importable:
    python -m ml.evaluation.compare_models   # or adjust sys.path below

Test cases live in ml/evaluation/test_cases.json — a list of VerdictInput
objects. Results are written to ml/evaluation/results/<timestamp>/.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make the backend package importable when run from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "backend" / ".env")

from app.schemas import VerdictInput  # noqa: E402
from app.services.llm.registry import available_models, get_provider  # noqa: E402
from app.services.prompt import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "test_cases.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    cases = [VerdictInput(**c) for c in json.loads(CASES_FILE.read_text(encoding="utf-8"))]
    models = available_models()
    if not models:
        sys.exit("No models available — set API keys in backend/.env")

    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)

    for i, case in enumerate(cases):
        user_prompt = build_user_prompt(case)
        for model in models:
            key = model["key"]
            print(f"case {i} / {key} ...", flush=True)
            start = time.time()
            text = get_provider(key).generate(SYSTEM_PROMPT, user_prompt)
            elapsed = time.time() - start
            out = run_dir / f"case{i}_{key}.json"
            out.write_text(
                json.dumps(
                    {
                        "case": case.model_dump(),
                        "model": key,
                        "elapsed_seconds": round(elapsed, 1),
                        "output": text,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    print(f"Results in {run_dir}")


if __name__ == "__main__":
    main()
