"""Landing page: what the engine is, the headline numbers, and the finding from each phase."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import streamlit as st  # noqa: E402

from common import (  # noqa: E402
    DEPLOYED_R2,
    DEPLOYED_RMSE,
    MODEL_SCORES,
    N_DOE_RUNS,
    N_FEATURES,
    N_LAYERS,
    N_WINES,
    TAGLINE,
    download_csv,
    provenance,
)

st.markdown(f"**{TAGLINE}.**")
st.markdown(
    "Six models, one bottle. Simulate a fermentation, test whether a tank is really one vessel, "
    "optimise the process, predict sensory quality from lab chemistry, rule out wines that could "
    "not legally or physically exist, then search for a better version of a wine that already "
    "does. Every page is something to try: move the inputs and the models re-run on your numbers."
)

with st.container(horizontal=True):
    st.metric(
        "Wines modelled",
        f"{N_WINES:,}",
        border=True,
        help="Real Portuguese white wines from the UCI dataset, duplicate rows removed.",
    )
    st.metric(
        "Lab measurements per wine",
        N_FEATURES,
        border=True,
        help="The 11 physico-chemical inputs to the quality model.",
    )
    st.metric(
        "Quality model R²",
        f"{DEPLOYED_R2:.2f}",
        border=True,
        help="Held-out test set, deployed Random Forest.",
    )
    st.metric(
        "Quality model RMSE",
        f"{DEPLOYED_RMSE:.2f}",
        border=True,
        help="Sensory points, on the 0–10 scale used by the tasting panel.",
    )
    st.metric(
        "Virtual experiments",
        N_DOE_RUNS,
        border=True,
        help="Box–Behnken runs on the fermentation twin, three centre points included.",
    )
    st.metric(
        "Feasibility layers",
        f"{N_LAYERS}",
        border=True,
        help="Legal, observed-range, typicality and physical-density checks.",
    )

st.subheader("The chain, and what each stage found", icon=":material/account_tree:")

STAGES = [
    (
        ":material/science:",
        "Fermentation twin",
        "A five-equation kinetic model of a white-wine fermentation, solved as an ODE system.",
        "Nitrogen, not sugar, is what runs out first and stalls the ferment.",
    ),
    (
        ":material/thermostat:",
        "Tank gradients",
        "Heat moves between horizontal layers of the tank instead of one uniform temperature.",
        "The gradient shifts the timing, not the destination: both layers land on the same ethanol.",
    ),
    (
        ":material/tune:",
        "Process optimisation",
        f"{N_DOE_RUNS} designed runs, second-order response surfaces, and a desirability optimum.",
        "The optimum sits on the boundary — warmer is better inside this window, not optimal at it.",
    ),
    (
        ":material/analytics:",
        "Quality model",
        f"A Random Forest maps {N_FEATURES} lab measurements to the panel's median sensory score.",
        "Alcohol dominates; the chemistry–quality link is real but non-linear and far from complete.",
    ),
    (
        ":material/fact_check:",
        "Feasibility",
        "Four layers reject wines that are illegal, atypical or physically impossible.",
        "Density is not a free parameter: it is fixed by alcohol and sugar to within ±3σ.",
    ),
    (
        ":material/auto_fix_high:",
        "Reformulation",
        "A constrained search for a nearby chemistry with a higher predicted score.",
        "The gap between scenarios is the finding: most of the gain is unreachable from the cellar.",
    ),
]

for row_start in (0, 3):
    cols = st.columns(3)
    for col, (icon, name, what, finding) in zip(
        cols, STAGES[row_start : row_start + 3]
    ):
        with col.container(border=True, height=212):
            st.markdown(f"##### {icon} {name}")
            st.caption(what)
            st.markdown(finding)

st.subheader("How to read these numbers", icon=":material/info:")
left, right = st.columns([3, 2])

with left.container(border=True):
    st.markdown(
        """
Three limits shape every number on these pages.

- **The quality model explains about 40 % of the variance** in human sensory scores, and that is
  close to the ceiling of this dataset rather than a tuning failure. Most of what a panel
  responds to is not in 11 lab measurements. Predicted scores rank wines; they do not promise
  points.
- **The model is trained on correlation, not intervention.** A predicted gain assumes the
  chemistry–quality relationship still holds when you deliberately move the chemistry. The
  feasibility layers keep the search inside the region where that assumption is least unsafe.
- **The process optimum is a window, not a point.** Temperature lands on the edge of the design
  range, and the desirability function carries no aroma term — which is exactly what a cooler
  white-wine ferment is for.
        """
    )

with right.container(border=True):
    st.markdown("**Candidate quality models, held-out test set**")
    st.dataframe(
        MODEL_SCORES,
        hide_index=True,
        width="stretch",
        alt="Test RMSE and R² for the Ridge, Random Forest and XGBoost candidates",
    )
    st.caption(
        "Trees beat the linear baseline by more than fold-to-fold noise, so the Random Forest "
        "ships. XGBoost matches it and adds no interpretability."
    )
    download_csv(
        MODEL_SCORES, "Model comparison (CSV)", "model_comparison.csv", key="dl_models"
    )

provenance(
    "Wine chemistry and sensory scores: UCI white-wine quality dataset. Process results: the "
    "repository's own simulator and R response-surface fits. Every model run is tracked in "
    "MLflow and reproduced by the notebooks in notebooks/."
)
