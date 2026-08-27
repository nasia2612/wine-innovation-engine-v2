"""Run the Phase 6 reformulation search for a sample of wines and log each run to MLflow.

The Streamlit Phase 6 page reads these runs from the `phase6_reformulation` experiment, so a
wine shows up in the app once this script has searched it.

    venv/bin/python scripts/run_phase6_batch.py --per-quality 6 --workers 6

Each search makes tens of thousands of model calls, so the script parallelises across wines and
runs both scenarios. Wines and scenarios that already have a finished run are skipped, so the
script can be restarted safely.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.modeling.constraints import load_artifacts  # noqa: E402
from src.modeling.reformulate import log_reformulation_run, reformulate  # noqa: E402

MLFLOW_DB = REPO_ROOT / "notebooks" / "mlflow.db"
PHASE4_RUN_ID = "25bd827ce16d409480c09b0dd37626ea"
RAW_CSV = REPO_ROOT / "data" / "raw" / "winequality-white (1).csv"
EXPERIMENT = "phase6_reformulation"
SCENARIOS = ["vintage_design", "cellar_intervention"]
ANCHOR_WINE = 536  # the wine the README discusses; always included


def _worker_setup():
    """Load the model and artifacts once per worker process."""
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    model = mlflow.sklearn.load_model("models:/wine_quality_rf/1")
    # One thread per call: a single-row predict otherwise spends most of its time on
    # thread-pool overhead. Predictions are identical either way.
    model.set_params(n_jobs=1)
    artifacts = load_artifacts(PHASE4_RUN_ID)
    return model, artifacts, artifacts["mahal"]["feature_order"]


def search(wine_index: int, scenario: str, lam: float):
    model, artifacts, feature_order = _worker_setup()
    wines = pd.read_csv(RAW_CSV, sep=";")
    x0 = wines.iloc[wine_index][feature_order].values.astype(float)
    return reformulate(model, x0, feature_order, artifacts, scenario=scenario, lam=lam)


def pick_wines(per_quality: int, seed: int) -> list[int]:
    """Random sample of wines at each quality score 4-7, plus the anchor wine."""
    wines = pd.read_csv(RAW_CSV, sep=";")
    rng = np.random.default_rng(seed)
    chosen = [ANCHOR_WINE]
    for q in [4, 5, 6, 7]:
        idx = wines.index[wines["quality"] == q].to_numpy()
        chosen += list(rng.choice(idx, size=min(per_quality, len(idx)), replace=False))
    return sorted({int(i) for i in chosen})


def already_done(client, exp_id, wine_index: int, scenario: str) -> bool:
    runs = client.search_runs(
        [exp_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"AND tags.scenario = '{scenario}' AND tags.wine_index = '{wine_index}'"
        ),
        max_results=1,
    )
    return len(runs) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-quality", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lam", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)

    wines = pick_wines(args.per_quality, args.seed)
    jobs = [
        (w, s)
        for w in wines
        for s in SCENARIOS
        if not already_done(client, exp_id, w, s)
    ]
    print(f"{len(wines)} wines selected, {len(jobs)} searches to run", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(search, w, s, args.lam): (w, s) for w, s in jobs}
        for fut in as_completed(futures):
            w, s = futures[fut]
            result = fut.result()
            # Log from the parent process: MLflow's SQLite store is written by one process.
            log_reformulation_run(client, result, w)
            gain = result.quality_after - result.quality_before
            print(
                f"wine {w:5d} {s:22s} gain {gain:+.3f} success {result.de_success}",
                flush=True,
            )


if __name__ == "__main__":
    main()
