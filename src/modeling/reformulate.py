"""Reformulation: constrained search over a wine's physicochemistry to raise predicted quality.

Implements algorithmic recourse (Karimi et al., 2020) over a trained quality model: search for
a feature vector that maximises predicted quality while staying inside four feasibility layers
(src.modeling.constraints.check) and penalising distance from the starting wine.

Two scenarios:
- "vintage_design": every feature is free to move (explores the whole feasible design space).
- "cellar_intervention": only lever features (free SO2, sulphates, citric acid by default) move;
  everything else is locked at the baseline wine's values, approximating what a winemaker can
  actually adjust post-fermentation.

Promoted out of notebooks/06_reformulation_optimizer.ipynb so the Streamlit app (Phase 6) and any
future caller can run a reformulation without re-deriving the DE setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, differential_evolution

from src.modeling.constraints import check

DEFAULT_ROLES = {
    "lever": ["free sulfur dioxide", "sulphates", "citric acid"],
    "semi": ["fixed acidity", "pH", "residual sugar"],
    "outcome": ["total sulfur dioxide", "density", "volatile acidity", "alcohol"],
    "context": ["chlorides"],
}


def build_bounds(feature_order, bounds_artifact):
    """L1 box bounds (per-feature 1st/99th percentile) -> (bounds list, scale array).

    scale = hi - lo per feature, used to normalise proximity cost across features with
    different units/ranges.
    """
    bounds = []
    for f in feature_order:
        lo, hi = bounds_artifact[f]
        assert lo < hi, f"empty interval for {f}"
        bounds.append((lo, hi))
    scale = np.array([hi - lo for lo, hi in bounds])
    assert (scale > 0).all(), "zero or negative scale"
    return bounds, scale


def lock_non_levers(bounds, feature_order, x0, levers):
    """Collapse every non-lever feature's bound to a single point at its baseline value."""
    x0 = np.asarray(x0, dtype=float)
    locked = []
    for i, feat in enumerate(feature_order):
        if feat in levers:
            locked.append(bounds[i])
        else:
            v = float(x0[i])
            locked.append((v, v))
    return locked


def mahalanobis_d2(x, mean, precision):
    diff = x - mean
    return diff @ precision @ diff


def build_constraints(feature_order, artifacts):
    """NonlinearConstraint objects for the DE search: L2 Mahalanobis typicality, L3 density-law
    band, L0 legal total-SO2 cap (sugar-dependent), and free SO2 <= total SO2.
    """
    mahal = artifacts["mahal"]
    density_band = artifacts["density_band"]
    so2_legal = artifacts["legal"]["total sulfur dioxide"]

    mean = np.array(mahal["location_"])
    precision = np.array(mahal["precision_"])
    nlc_mahal = NonlinearConstraint(
        lambda x: mahalanobis_d2(x, mean, precision),
        lb=-np.inf,
        ub=mahal["threshold_d2"],
    )

    i_alc = feature_order.index("alcohol")
    i_sugar = feature_order.index("residual sugar")
    i_dens = feature_order.index("density")
    c = density_band["coef"]
    sigma = density_band["sigma"]
    k = density_band["k"]

    def density_z(x):
        predicted = (
            c["const"] + c["alcohol"] * x[i_alc] + c["residual sugar"] * x[i_sugar]
        )
        return abs(x[i_dens] - predicted) / sigma

    nlc_density = NonlinearConstraint(density_z, lb=-np.inf, ub=k)

    i_tso2 = feature_order.index("total sulfur dioxide")
    i_fso2 = feature_order.index("free sulfur dioxide")

    def so2_margin(x):
        cap = (
            so2_legal["cap_high_sugar"]
            if x[i_sugar] >= so2_legal["sugar_threshold_gL"]
            else so2_legal["cap_low_sugar"]
        )
        return cap - x[i_tso2]

    nlc_so2 = NonlinearConstraint(so2_margin, lb=0, ub=np.inf)

    def free_le_total(x):
        return x[i_tso2] - x[i_fso2]

    nlc_free = NonlinearConstraint(free_le_total, lb=0, ub=np.inf)

    return [nlc_mahal, nlc_density, nlc_so2, nlc_free]


def make_objective(model, x0, scale, lam):
    """-predicted_quality + lam * L1 proximity cost (DE always minimises)."""
    x0 = np.asarray(x0, dtype=float)

    def objective(x):
        q = model.predict(x.reshape(1, -1))[0]
        cost = np.sum(np.abs(x - x0) / scale)
        return -q + lam * cost

    return objective


@dataclass
class ReformulationResult:
    scenario: str
    x0: np.ndarray
    x_star: np.ndarray
    feature_order: list
    scale: np.ndarray
    quality_before: float
    quality_after: float
    proximity_cost: float
    lam: float
    de_success: bool
    de_message: str
    de_nfev: int
    feasibility: dict = field(repr=False)

    @property
    def delta_table(self) -> pd.DataFrame:
        delta = self.x_star - self.x0
        return pd.DataFrame(
            {
                "before": self.x0,
                "after": self.x_star,
                "delta": delta,
                "pct_range": delta / self.scale,
            },
            index=self.feature_order,
        ).sort_values("pct_range", key=np.abs, ascending=False)


def reformulate(
    model,
    x0,
    feature_order,
    artifacts,
    scenario="vintage_design",
    levers=None,
    lam=1.5,
    seed=42,
    maxiter=200,
    popsize=30,
) -> ReformulationResult:
    """Run the constrained differential-evolution search for one wine.

    `artifacts` is the dict returned by src.modeling.constraints.load_artifacts (keys: bounds,
    mahal, density_band, legal). `x0` must already be ordered to match `feature_order`.
    """
    if scenario not in {"vintage_design", "cellar_intervention"}:
        raise ValueError(f"unknown scenario: {scenario!r}")

    x0 = np.asarray(x0, dtype=float)
    bounds, scale = build_bounds(feature_order, artifacts["bounds"]["bounds"])

    if scenario == "cellar_intervention":
        levers = set(levers or DEFAULT_ROLES["lever"])
        bounds = lock_non_levers(bounds, feature_order, x0, levers)

    constraints = build_constraints(feature_order, artifacts)
    objective = make_objective(model, x0, scale, lam)

    res = differential_evolution(
        objective,
        bounds=bounds,
        constraints=constraints,
        seed=seed,
        polish=False,
        maxiter=maxiter,
        popsize=popsize,
        init="latinhypercube",
    )

    x_star = res.x
    q0 = float(model.predict(x0.reshape(1, -1))[0])
    q_star = float(model.predict(x_star.reshape(1, -1))[0])
    cost = float(np.sum(np.abs(x_star - x0) / scale))

    feasibility = check(
        pd.Series(x_star, index=feature_order),
        artifacts["bounds"]["bounds"],
        artifacts["mahal"],
        artifacts["density_band"],
        artifacts["legal"],
    )

    return ReformulationResult(
        scenario=scenario,
        x0=x0,
        x_star=x_star,
        feature_order=list(feature_order),
        scale=scale,
        quality_before=q0,
        quality_after=q_star,
        proximity_cost=cost,
        lam=lam,
        de_success=bool(res.success),
        de_message=res.message,
        de_nfev=int(res.nfev),
        feasibility=feasibility,
    )


def log_reformulation_run(
    client,
    result: ReformulationResult,
    wine_index,
    experiment_name="phase6_reformulation",
):
    """Log a ReformulationResult to MLflow the same way both scenarios were logged in the
    original notebook (one run per scenario, tagged by wine_index)."""
    exp = client.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp else client.create_experiment(experiment_name)

    run = client.create_run(
        exp_id,
        tags={
            "role": "phase6_optimization",
            "scenario": result.scenario,
            "wine_index": str(wine_index),
        },
    )
    rid = run.info.run_id

    for k, v in {
        "quality_baseline": result.quality_before,
        "quality_final": result.quality_after,
        "quality_gain": result.quality_after - result.quality_before,
        "proximity_cost": result.proximity_cost,
        "penalty": result.lam * result.proximity_cost,
        "objective_check": -result.quality_after + result.lam * result.proximity_cost,
        "de_success": int(result.de_success),
        "de_nfev": result.de_nfev,
    }.items():
        client.log_metric(rid, k, float(v))

    client.log_dict(
        rid, result.delta_table.reset_index().to_dict(orient="list"), "delta_table.json"
    )
    client.log_dict(
        rid,
        dict(zip(result.feature_order, np.asarray(result.x_star).tolist())),
        "solution_vector.json",
    )
    client.log_dict(rid, result.feasibility, "constraint_audit.json")
    client.set_terminated(rid)
    return rid
