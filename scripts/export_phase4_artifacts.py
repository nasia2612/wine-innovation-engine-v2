"""Export the Phase 4 model and constraint artifacts out of MLflow into results/.

The app used to read both from the local MLflow store under notebooks/mlruns/, which is not in
version control. That made a fresh clone unable to render the Feasibility and Reformulation
pages. This script copies everything those pages need into results/phase4_export/, which is
small enough to commit, so the app runs from a clone with no MLflow store present.

    venv/bin/python scripts/export_phase4_artifacts.py

Re-run it after retraining the quality model. The app prefers these files and falls back to
MLflow when they are absent, so deleting the directory restores the old behaviour.
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import sklearn  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.modeling.constraints import load_artifacts  # noqa: E402

MLFLOW_DB = REPO_ROOT / "notebooks" / "mlflow.db"
PHASE4_RUN_ID = "25bd827ce16d409480c09b0dd37626ea"
MODEL_URI = "models:/wine_quality_rf/1"
EXPORT_DIR = REPO_ROOT / "results" / "phase4_export"
PHASE6_EXPORT = REPO_ROOT / "results" / "phase6_export" / "reformulations.json"
CONSTRAINT_NAMES = ("bounds", "mahal", "density_band", "legal")
SCENARIOS = ("vintage_design", "cellar_intervention")

# compress=3 takes the 300-tree forest from 25 MB to about 5.6 MB, which is comfortable in git
# and loads faster than MLflow's own loader.
COMPRESS = 3


def main() -> int:
    if not MLFLOW_DB.exists():
        print(
            f"error: no MLflow store at {MLFLOW_DB}; nothing to export from.",
            file=sys.stderr,
        )
        return 1

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    client = MlflowClient()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    constraints = load_artifacts(PHASE4_RUN_ID, client=client)
    for name in CONSTRAINT_NAMES:
        path = EXPORT_DIR / f"{name}.json"
        path.write_text(json.dumps(constraints[name], indent=2, sort_keys=True) + "\n")
        print(
            f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1024:.1f} KiB)"
        )

    model = mlflow.sklearn.load_model(MODEL_URI)
    model_path = EXPORT_DIR / "wine_quality_rf.joblib"
    joblib.dump(model, model_path, compress=COMPRESS)
    print(
        f"wrote {model_path.relative_to(REPO_ROOT)} "
        f"({model_path.stat().st_size / 1e6:.1f} MB)"
    )

    run = client.get_run(PHASE4_RUN_ID)
    metadata = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run_id": PHASE4_RUN_ID,
        "source_model_uri": MODEL_URI,
        "model_class": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", None),
        "feature_order": constraints["mahal"]["feature_order"],
        # Pinned so a loader can warn instead of unpickling into a silently different sklearn.
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "python_version": platform.python_version(),
        "source_run_metrics": {k: v for k, v in sorted(run.data.metrics.items())},
    }
    meta_path = EXPORT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {meta_path.relative_to(REPO_ROOT)}")

    # Fail loudly here rather than in the app: a reloaded model that disagrees with the one in
    # MLflow would make every predicted quality on the Feasibility page wrong.
    import numpy as np

    reloaded = joblib.load(model_path)
    probe = np.array(
        [constraints["bounds"]["bounds"][f][0] for f in metadata["feature_order"]]
    )
    a = float(model.predict([probe])[0])
    b = float(reloaded.predict([probe])[0])
    if not np.isclose(a, b):
        print(
            f"error: reloaded model disagrees with MLflow ({a} vs {b}).",
            file=sys.stderr,
        )
        return 1
    print(f"verified: reloaded model reproduces MLflow prediction ({a:.6f})")

    export_phase6(client)
    return 0


def export_phase6(client):
    """Flatten the finished phase6_reformulation runs into one committed JSON file.

    The Reformulation page reads tens of runs' metrics and two artifacts each. Without this the
    page has nothing to show in a clone, because the runs live only in the local MLflow store.
    """
    exp = client.get_experiment_by_name("phase6_reformulation")
    if exp is None:
        print("no phase6_reformulation experiment; skipping Phase 6 export")
        return

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=5000,
    )
    # Newest run wins per (wine, scenario), matching what the app showed when it read MLflow.
    results: dict[str, dict] = {}
    for run in runs:
        tags = run.data.tags
        wine, scenario = tags.get("wine_index"), tags.get("scenario")
        if wine is None or scenario not in SCENARIOS:
            continue
        key = f"{int(wine)}/{scenario}"
        if key in results:
            continue
        files = {a.path for a in client.list_artifacts(run.info.run_id)}
        if not {"delta_table.json", "constraint_audit.json"} <= files:
            continue
        rid = run.info.run_id
        results[key] = {
            "run_id": rid,
            "metrics": dict(run.data.metrics),
            "delta": mlflow.artifacts.load_dict(f"runs:/{rid}/delta_table.json"),
            "audit": mlflow.artifacts.load_dict(f"runs:/{rid}/constraint_audit.json"),
        }

    complete = sorted(
        {int(k.split("/")[0]) for k in results}
        - {
            int(k.split("/")[0])
            for k in results
            if not all(f"{k.split('/')[0]}/{s}" in results for s in SCENARIOS)
        }
    )
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete_wines": complete,
        "runs": results,
    }
    PHASE6_EXPORT.parent.mkdir(parents=True, exist_ok=True)
    PHASE6_EXPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {PHASE6_EXPORT.relative_to(REPO_ROOT)} "
        f"({PHASE6_EXPORT.stat().st_size / 1024:.1f} KiB, {len(complete)} wines with both "
        f"scenarios)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
