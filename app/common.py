"""Shared setup for the Streamlit app: repo imports, MLflow connection, cached loaders.

Everything in here is used by more than one page: the branding strings the shell and the
overview both show, the headline model numbers quoted in several places, and the cached
loaders that keep MLflow and the quality model off the critical path of every rerun.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.modeling.constraints import (  # noqa: E402
    EXPORT_DIR,
    load_constraints,
    load_quality_model,
)

APP_NAME = "Wine Innovation Engine"
TAGLINE = "Computational product development for white wine"

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = ASSETS / "logo.svg"
MARK = ASSETS / "mark.svg"

MLFLOW_DB = REPO_ROOT / "notebooks" / "mlflow.db"
PHASE4_RUN_ID = "25bd827ce16d409480c09b0dd37626ea"
RAW_CSV = REPO_ROOT / "data" / "raw" / "winequality-white (1).csv"
# Phase 6 searches flattened out of MLflow by scripts/export_phase4_artifacts.py, so the
# Reformulation page has something to show in a clone.
PHASE6_EXPORT = REPO_ROOT / "results" / "phase6_export" / "reformulations.json"

# Held-out test-set scores for the Phase 4 candidates, as logged to MLflow and quoted in the
# README. Kept here because the overview and the quality-model page both report them.
MODEL_SCORES = pd.DataFrame(
    {
        "model": ["Ridge (linear baseline)", "Random Forest (deployed)", "XGBoost"],
        "test RMSE": [0.745, 0.691, 0.692],
        "test R²": [0.27, 0.398, 0.396],
    }
)

N_WINES = 3961
N_DOE_RUNS = 15
N_FEATURES = 11
N_LAYERS = 4
DEPLOYED_RMSE = 0.691
DEPLOYED_R2 = 0.398


def provenance(text: str) -> None:
    """One-line note on where a page's numbers come from. Clients ask; the answer is on screen."""
    st.caption(f":material/database: {text}")


def download_csv(
    df: pd.DataFrame, label: str, file_name: str, key: str | None = None
) -> None:
    """Offer a table as a CSV download, so a reviewer can take the numbers away."""
    st.download_button(
        label,
        df.to_csv(index=False).encode(),
        file_name=file_name,
        mime="text/csv",
        icon=":material/download:",
        key=key,
    )


@st.cache_data(show_spinner=False, ttl=60)
def _phase6_export() -> dict:
    """The committed Phase 6 results, or {} when the export has not been written."""
    if not PHASE6_EXPORT.exists():
        return {}
    with open(PHASE6_EXPORT) as f:
        return json.load(f)


@st.cache_data(show_spinner=False, ttl=60)
def list_searched_wines() -> list[int]:
    """Wine rows with both Phase 6 scenarios finished and loadable, sorted by row number.

    A wine is listed only once both scenarios have their artifacts on disk, so the page never
    shows half a result. Checking the artifacts matters rather than trusting run status: a run
    can finish without writing them, and listing such a wine leaves an empty card on screen.
    """
    export = _phase6_export()
    if export:
        return list(export["complete_wines"])

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("phase6_reformulation")
    if exp is None:
        return []
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=5000,
    )
    done: dict[int, set[str]] = {}
    for r in runs:
        if "wine_index" not in r.data.tags:
            continue
        files = {a.path for a in client.list_artifacts(r.info.run_id)}
        if {"delta_table.json", "constraint_audit.json"} <= files:
            done.setdefault(int(r.data.tags["wine_index"]), set()).add(
                r.data.tags.get("scenario")
            )
    return sorted(
        w
        for w, scenarios in done.items()
        if {"vintage_design", "cellar_intervention"} <= scenarios
    )


@st.cache_data(show_spinner=False, ttl=60)
def load_saved_reformulation(wine_index: int, scenario: str):
    """Latest finished Phase 6 MLflow run for (wine, scenario), or None if there is none.

    These are the runs the README quotes, so the page shows the same numbers without
    re-running the optimiser (a single search makes tens of thousands of model calls).
    """
    export = _phase6_export()
    if export:
        saved = export["runs"].get(f"{wine_index}/{scenario}")
        if saved is None:
            return None
        return {
            "metrics": saved["metrics"],
            "delta": pd.DataFrame(saved["delta"]),
            "audit": saved["audit"],
            "run_id": saved["run_id"],
        }

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("phase6_reformulation")
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"AND tags.scenario = '{scenario}' AND tags.wine_index = '{wine_index}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    for run in runs:
        files = {a.path for a in client.list_artifacts(run.info.run_id)}
        if {"delta_table.json", "constraint_audit.json"} <= files:
            rid = run.info.run_id
            return {
                "metrics": run.data.metrics,
                "delta": pd.DataFrame(
                    mlflow.artifacts.load_dict(f"runs:/{rid}/delta_table.json")
                ),
                "audit": mlflow.artifacts.load_dict(
                    f"runs:/{rid}/constraint_audit.json"
                ),
                "run_id": rid,
            }
    return None


@st.cache_resource(show_spinner="Loading quality model and constraints...")
def load_phase6():
    """Return (model, artifacts, feature_order, wines) for the Phase 5 and 6 pages.

    Both the model and the constraints come from results/phase4_export/, which is committed, so
    these pages work in a fresh clone with no MLflow store present. The loaders fall back to
    MLflow when the export is missing.
    """
    if not EXPORT_DIR.exists():
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    model = load_quality_model()
    artifacts = load_constraints(run_id=PHASE4_RUN_ID)
    feature_order = artifacts["mahal"]["feature_order"]
    wines = pd.read_csv(RAW_CSV, sep=";")
    return model, artifacts, feature_order, wines
