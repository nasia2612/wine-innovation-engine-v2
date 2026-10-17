"""Phase 4: the sensory-quality model and its SHAP explanation, from the saved test-set SHAP values
in results/phase4_shap/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (  # noqa: E402
    MODEL_SCORES,
    N_FEATURES,
    N_WINES,
    REPO_ROOT,
    download_csv,
    provenance,
)

SHAP_DIR = REPO_ROOT / "results" / "phase4_shap"

# Direction colours. These are the reserved status steps, not series colours: they clear 3:1 on
# both the light and dark chart surfaces, and the sign of the axis carries the same information.
RAISES, LOWERS = "#0CA30C", "#D03B3B"


@st.cache_data(show_spinner=False)
def load_shap():
    X = pd.read_parquet(SHAP_DIR / "X_test.parquet")
    y = pd.read_parquet(SHAP_DIR / "y_test.parquet")["quality"]
    values = pd.DataFrame(
        np.load(SHAP_DIR / "shap_values.npy"), columns=X.columns, index=X.index
    )
    base = float(np.load(SHAP_DIR / "shap_base.npy").ravel()[0])
    return X, y, values, base


X, y, shap_values, base = load_shap()

st.markdown(
    f"A Random Forest trained on {N_WINES:,} real Portuguese white wines predicts the median "
    f"sensory score of a tasting panel from {N_FEATURES} routine lab measurements. SHAP then "
    f"splits every individual prediction into one contribution per measurement, so the model can "
    f"be questioned wine by wine rather than trusted wholesale. Everything below is computed on "
    f"the {len(X)} held-out test wines the model never saw."
)

importance = shap_values.abs().mean().sort_values(ascending=False).reset_index()
importance.columns = ["feature", "mean |SHAP|"]

left, right = st.columns([2, 3])
with left.container(border=True):
    st.markdown("**Candidate models, held-out test set**")
    st.dataframe(
        MODEL_SCORES,
        hide_index=True,
        width="stretch",
        alt="Test RMSE and R² for the Ridge, Random Forest and XGBoost candidates",
    )
    st.caption(
        "Trees beat the linear baseline by more than fold-to-fold noise, so the chemistry–quality "
        "link is genuinely non-linear. R² ≈ 0.40 is close to the ceiling of this data, not a "
        "tuning failure: most of the variance in human scores is not in these 11 numbers."
    )
    download_csv(
        MODEL_SCORES, "Model comparison (CSV)", "model_comparison.csv", key="dl_scores"
    )

with right.container(border=True):
    st.altair_chart(
        alt.Chart(importance)
        .mark_bar(cornerRadiusEnd=4, height=14)
        .encode(
            x=alt.X("mean |SHAP|:Q", title="Mean |SHAP| (quality points)"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("mean |SHAP|", format=".3f")],
        )
        .properties(
            height=330, title="What the model leans on, averaged over the test set"
        ),
        alt="Bar chart of mean absolute SHAP value per feature",
    )
    st.caption(
        "Average size of each measurement's effect on a prediction, in sensory points. Alcohol "
        "carries roughly as much as the next three measurements combined."
    )

st.subheader("How one measurement moves the score", icon=":material/scatter_plot:")
feature = st.selectbox("Measurement", importance["feature"].tolist())
dep = pd.DataFrame(
    {feature: X[feature], "SHAP": shap_values[feature], "alcohol": X["alcohol"]}
)
with st.container(border=True):
    st.altair_chart(
        alt.Chart(dep)
        .mark_circle(size=42, opacity=0.65, stroke="white", strokeWidth=0.4)
        .encode(
            x=alt.X(f"{feature}:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("SHAP:Q", title="Contribution to predicted quality (points)"),
            color=alt.Color("alcohol:Q", title="Alcohol (% vol)"),
            tooltip=[
                alt.Tooltip(f"{feature}:Q", format=".3f"),
                alt.Tooltip("SHAP:Q", format=".3f", title="contribution"),
                alt.Tooltip("alcohol:Q", format=".1f"),
            ],
        )
        .properties(height=330),
        alt=f"SHAP dependence of {feature}, coloured by alcohol",
    )
    notes = {
        "alcohol": "Alcohol dominates, and monotonically: higher alcohol pushes the score up "
        "across the whole observed range.",
        "free sulfur dioxide": "Strongly asymmetric. Very low free SO₂ drags scores down hard — "
        "an unprotected, oxidation-prone wine — while high values simply level off.",
        "volatile acidity": "High volatile acidity, the vinegar fault, pushes scores down.",
        "residual sugar": "Weak on its own, and largely read through its interaction with "
        "alcohol, which the colour of the points exposes.",
    }
    st.caption(
        notes.get(
            feature,
            "Each point is one test wine: its measured value against that measurement's "
            "contribution to its predicted score. Colour shows alcohol, the strongest "
            "interacting measurement.",
        )
    )

st.subheader("Explain a single wine", icon=":material/troubleshoot:")
st.markdown(
    "Not which measurement matters on average, but why *this* wine scored what it scored. Pick a "
    "wine and take its prediction apart."
)
row = st.number_input(
    f"Test-set row (0 to {len(X) - 1})", 0, len(X) - 1, 0, help="Held-out wines only."
)
contrib = shap_values.iloc[row]
pred = base + contrib.sum()
with st.container(horizontal=True):
    st.metric(
        "Average prediction",
        f"{base:.2f}",
        border=True,
        help="The model's output before any of this wine's measurements are taken into account.",
    )
    st.metric(
        "Prediction for this wine",
        f"{pred:.2f}",
        f"{pred - base:+.2f}",
        border=True,
        delta_description="vs. the average wine",
    )
    st.metric("Actual panel score", f"{int(y.iloc[row])}", border=True)
    st.metric(
        "Residual",
        f"{pred - y.iloc[row]:+.2f}",
        border=True,
        help="Prediction minus what the panel actually gave. Typical error is about 0.69 points.",
    )

wf = (
    pd.DataFrame(
        {"feature": contrib.index, "SHAP": contrib.values, "value": X.iloc[row].values}
    )
    .assign(
        direction=lambda d: np.where(d["SHAP"] >= 0, "raises score", "lowers score")
    )
    .sort_values("SHAP", key=np.abs, ascending=False)
)
with st.container(border=True):
    st.altair_chart(
        alt.Chart(wf)
        .mark_bar(cornerRadiusEnd=4, height=14)
        .encode(
            x=alt.X("SHAP:Q", title="Contribution to predicted quality (points)"),
            y=alt.Y("feature:N", sort=wf["feature"].tolist(), title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["raises score", "lowers score"], range=[RAISES, LOWERS]
                ),
                title=None,
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                "feature",
                alt.Tooltip("value", format=".3f", title="measured value"),
                alt.Tooltip("SHAP", format=".3f", title="contribution"),
            ],
        )
        .properties(height=330, title=f"Why wine #{row} scores {pred:.2f}"),
        alt="Per-feature SHAP contributions for the selected wine",
    )
    st.caption(
        f"The average prediction ({base:.2f}) plus every bar equals this wine's prediction "
        f"({pred:.2f}). Bars are ordered by size of effect, not by sign."
    )
    download_csv(
        wf[["feature", "value", "SHAP", "direction"]].round(4),
        "This explanation (CSV)",
        f"shap_wine_{row}.csv",
        key="dl_wf",
    )

provenance(
    "Data: UCI white-wine quality dataset, duplicate rows removed. Model, SHAP values and test "
    "metrics are logged to MLflow by notebooks/04_*; the saved arrays live in "
    "results/phase4_shap/."
)
