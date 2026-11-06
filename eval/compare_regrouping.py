"""Does a stronger model regroup complaints better? (task B13)

    MIA_DATA_DIR=<scratch dir> uv run python eval/compare_regrouping.py <run.json> ... \\
        [--model gemini/<id>:<thinking> ...]

Runs the regrouping step several times per model on each run's complaints and prints B-cubed F1
against gold/complaint_themes_*.json. Prompts vary slightly so each try gets its own answer. Use a
scratch data directory, with `models` linked from data/, to keep these answers out of the cache.
Each try is a real request, and Flash models allow only about 20 a day.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import tune_clustering as tc  # noqa: E402

from mia import runtime  # noqa: E402
from mia.analysis import clustering  # noqa: E402
from mia.rag.dense import embed_texts  # noqa: E402
from mia.schemas import Complaint  # noqa: E402

CONFIGS = [
    ("gemini/gemini-3.5-flash-lite", "low"),
    ("gemini/gemini-3.6-flash", "medium"),
    ("gemini/gemini-3.8-flash", "medium"),
]
TRIES = 3


def _on_model(extract, model: str, thinking: str, attempt: int):
    def varied(prompt, *, schema, system, **_):
        return extract(
            f"{prompt}\n(try {attempt})",
            schema=schema,
            system=system,
            model=model,
            thinking=thinking,
        )

    return varied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--model", action="append", help="gemini/<id>:<thinking>; default: all")
    args = parser.parse_args()
    configs = [tuple(m.rsplit(":", 1)) for m in args.model] if args.model else CONFIGS
    runtime.configure()
    extract = runtime.llm().extract
    for run in args.runs:
        report = json.loads(Path(run).read_text(encoding="utf-8"))
        complaints = [Complaint.model_validate(c) for c in report["complaints"]]
        universe = [c.summary for c in complaints]
        vectors = np.array(embed_texts(universe), dtype=float)
        indices = clustering.agglomerative(vectors, runtime.settings().cluster_distance)
        seeds = [[complaints[i] for i in g] for g in indices]
        print(
            f"\n{run}: seeds alone",
            tc.describe([[universe[i] for i in g] for g in indices], universe),
        )
        for model, thinking in configs:
            for attempt in range(TRIES):
                runtime.llm().extract = _on_model(extract, model, thinking, attempt)
                warned = len(runtime.warnings())
                refined = clustering.refine(seeds)
                answered = len(runtime.warnings()) == warned
                groups = [[c.summary for c in members] for _, _, members in refined]
                print(
                    model,
                    attempt,
                    "answered" if answered else "no answer",
                    tc.describe(groups, universe),
                )


if __name__ == "__main__":
    main()
