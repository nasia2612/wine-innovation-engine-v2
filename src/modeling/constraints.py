import json
import warnings

import numpy as np

from pathlib import Path

# mlflow is imported inside the functions that need it. It costs ~160 MB and ~1.7 s to
# import, and it is only ever the fallback: the committed export under results/ is what
# the app actually reads, so a deployed instance never pays for it.

REPO_ROOT = Path(__file__).resolve().parents[2]
# Committed copies of everything the app needs from Phase 4, written by
# scripts/export_phase4_artifacts.py. The MLflow store under notebooks/ is not in version
# control, so without these a fresh clone cannot score or screen a wine.
EXPORT_DIR = REPO_ROOT / "results" / "phase4_export"
CONSTRAINT_NAMES = ("bounds", "mahal", "density_band", "legal")


def load_artifacts(run_id, client=None):
    if client is None:  # if you dont provide a client, create one
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
    # dependency injection : the function is not responsible for creating the client, it just uses it. This makes the function more testable and flexible.
    out = {}
    for name in ["bounds", "mahal", "density_band", "legal"]:
        p = client.download_artifacts(
            run_id, f"constraints/{name}.json"
        )  # it downloads the artifact from mlflow and returns the local path to the downloaded file
        with open(p) as f:
            out[name] = json.load(f)  # it loads the json file and returns a dictionary

    return out


def _export_metadata(export_dir=EXPORT_DIR):
    meta = export_dir / "metadata.json"
    if not meta.exists():
        return {}
    with open(meta) as f:
        return json.load(f)


def load_constraints(run_id=None, client=None, export_dir=EXPORT_DIR):
    """The four constraint artifacts, read from the committed export when it is present.

    Falls back to `load_artifacts` against the local MLflow store, so deleting the export
    directory restores the previous behaviour and a retrained run can still be read directly.
    """
    missing = [n for n in CONSTRAINT_NAMES if not (export_dir / f"{n}.json").exists()]
    if missing:
        if run_id is None:
            raise FileNotFoundError(
                f"no exported constraints in {export_dir} (missing {', '.join(missing)}) and no "
                "run_id given. Run scripts/export_phase4_artifacts.py."
            )
        return load_artifacts(run_id, client=client)
    out = {}
    for name in CONSTRAINT_NAMES:
        with open(export_dir / f"{name}.json") as f:
            out[name] = json.load(f)
    return out


def load_quality_model(model_uri="models:/wine_quality_rf/1", export_dir=EXPORT_DIR):
    """The deployed quality model, from the committed export when it is present.

    The export is a joblib pickle, so it is only valid for the scikit-learn it was written by.
    The recorded version is compared against the running one and a mismatch warns rather than
    failing, because minor versions usually load fine and a hard error here would take the
    whole app down.
    """
    path = export_dir / "wine_quality_rf.joblib"
    if not path.exists():
        import mlflow.sklearn

        return mlflow.sklearn.load_model(model_uri)

    import joblib
    import sklearn

    expected = _export_metadata(export_dir).get("sklearn_version")
    if expected and expected != sklearn.__version__:
        warnings.warn(
            f"{path.name} was written by scikit-learn {expected} but {sklearn.__version__} is "
            "running. Re-run scripts/export_phase4_artifacts.py if predictions look wrong.",
            RuntimeWarning,
            stacklevel=2,
        )
    return joblib.load(path)


def check(x, bounds, mahal, density_band, legal):
    assert (
        list(x.index) == mahal["feature_order"]
    )  # we want them in the same order as the one used to compute the mahalanobis distance, otherwise the distance will be wrong
    v = x.values

    lo = np.array(
        [bounds[f][0] for f in x.index]
    )  # the 1st percentile of the feature f, for each feature in x.index
    hi = np.array(
        [bounds[f][1] for f in x.index]
    )  # the 99th percentile of the feature f, for each feature in x.index
    # location is the mean of each feature in the training set
    # the _ in location_ and precision_ is a convention used by sklearn to indicate that these attributes are estimated from the training data and are not part of the model parameters.

    diff = v - np.array(mahal["location_"])  #  how far from standard
    d2 = (
        diff @ np.array(mahal["precision_"]) @ diff
    )  # the precision of a measurement is the inverse of its variance: τ = 1/σ²

    c = density_band["coef"]
    pred = (
        c["const"]
        + c["alcohol"] * x["alcohol"]
        + c["residual sugar"] * x["residual sugar"]
    )
    band = abs(x["density"] - pred) / density_band["sigma"]

    so2_cap = (
        legal["total sulfur dioxide"]["cap_high_sugar"]
        if x["residual sugar"] >= 5.0
        else legal["total sulfur dioxide"]["cap_low_sugar"]
    )

    return {
        "L1_bounds": bool(((v >= lo) & (v <= hi)).all()),
        "L2_mahal": bool(d2 <= mahal["threshold_d2"]),
        "L3_density": bool(band <= density_band["k"]),
        "L0_legal": bool(
            x["total sulfur dioxide"] <= so2_cap
            and x["volatile acidity"] <= legal["volatile acidity"]["cap"]
        ),
        "d2": float(d2),
        "z_density": float(band),
    }


def setup_mlflow():
    import mlflow

    repo = Path(__file__).resolve().parents[2]
    db = repo / "notebooks" / "mlflow.db"
    assert db.exists(), f"tracking DB not found at {db}"
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    return repo
