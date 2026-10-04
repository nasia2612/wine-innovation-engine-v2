"""Phase 5: feasibility layers L0-L3 from src/modeling/constraints.py. Edit a wine's chemistry and
see which layers it passes, alongside the quality model's prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import load_phase6, provenance  # noqa: E402
from src.modeling.constraints import check  # noqa: E402

# The acceptable band is a status region, and the selected wine is a UI selection marker, so
# neither takes a series colour. Both clear 3:1 on the light and dark chart surfaces.
BAND, SELECTED, CLOUD = "#0CA30C", "#2A78D6", "#898781"

model, artifacts, feature_order, wines = load_phase6()
bounds = artifacts["bounds"]["bounds"]
mahal, density_band, legal = (
    artifacts["mahal"],
    artifacts["density_band"],
    artifacts["legal"],
)


def load_wine_into_inputs():
    row = wines.iloc[st.session_state.wine_row]
    for f in feature_order:
        st.session_state[f"feat_{f}"] = float(row[f])


if "feat_alcohol" not in st.session_state:
    st.session_state.wine_row = 536
    load_wine_into_inputs()

st.markdown(
    "The quality model will happily score any list of eleven numbers, including wines that could "
    "not legally be sold or could not physically exist. These four layers are what keep the "
    "reformulation search honest. Start from a real wine, change its chemistry, and watch which "
    "layers break."
)

with st.sidebar:
    st.subheader("Starting wine", icon=":material/tune:")
    st.number_input(
        f"Wine row (0 to {len(wines) - 1})",
        0,
        len(wines) - 1,
        key="wine_row",
        on_change=load_wine_into_inputs,
    )
    st.button(
        "Reset to this wine",
        on_click=load_wine_into_inputs,
        icon=":material/restart_alt:",
        width="stretch",
    )
    st.caption("Editing the chemistry below never changes the stored dataset.")

with st.form("chemistry", border=True):
    st.markdown("**Chemistry of the wine under test**")
    cols = st.columns(4)
    for i, f in enumerate(feature_order):
        lo, hi = bounds[f]
        cols[i % 4].number_input(
            f,
            key=f"feat_{f}",
            format="%.5f" if f == "density" else "%.3f",
            step=0.0005 if f == "density" else 0.1,
            help=f"Observed 1st–99th percentile: {lo:.4g} to {hi:.4g}",
        )
    st.form_submit_button(
        "Check this wine", icon=":material/fact_check:", type="primary"
    )

x = pd.Series({f: st.session_state[f"feat_{f}"] for f in feature_order})[feature_order]
result = check(x, bounds, mahal, density_band, legal)
quality = float(model.predict([x.values])[0])

sugar_high = x["residual sugar"] >= legal["total sulfur dioxide"]["sugar_threshold_gL"]
so2_cap = legal["total sulfur dioxide"][
    "cap_high_sugar" if sugar_high else "cap_low_sugar"
]
va_cap = legal["volatile acidity"]["cap"]
out_of_range = [f for f in feature_order if not bounds[f][0] <= x[f] <= bounds[f][1]]
c = density_band["coef"]
density_pred = (
    c["const"] + c["alcohol"] * x["alcohol"] + c["residual sugar"] * x["residual sugar"]
)

n_pass = sum(result[k] for k in ("L0_legal", "L1_bounds", "L2_mahal", "L3_density"))
with st.container(horizontal=True):
    st.metric(
        "Predicted quality",
        f"{quality:.2f}",
        border=True,
        help="What the Phase 4 Random Forest scores this chemistry. It does not know whether the "
        "wine is possible.",
    )
    st.metric(
        "Feasibility layers passed",
        f"{n_pass} / 4",
        border=True,
        help="A wine must clear all four to be a legitimate reformulation target.",
    )

if n_pass == 4:
    st.success(
        "This wine is legal, typical and physically consistent: a legitimate target for the "
        "reformulation search.",
        icon=":material/verified:",
    )
else:
    st.error(
        f"{4 - n_pass} of 4 layers rejected this wine, so its predicted quality is not a "
        f"usable number. The failing layer is named below.",
        icon=":material/block:",
    )

st.subheader("The four layers", icon=":material/rule:")
layers = [
    (
        "L0 · legal",
        result["L0_legal"],
        (
            f"Total SO₂ {x['total sulfur dioxide']:.0f} mg/L against a cap of {so2_cap:.0f} "
            f"mg/L (the cap depends on whether sugar reaches 5 g/L); volatile acidity "
            f"{x['volatile acidity']:.2f} against a cap of {va_cap:.2f} g/L. Limits from "
            f"Reg (EU) 2019/934."
        ),
    ),
    (
        "L1 · within observed ranges",
        result["L1_bounds"],
        (
            "Every value sits inside the 1st–99th percentile of the training wines."
            if not out_of_range
            else "Outside the observed range: " + ", ".join(out_of_range) + "."
        ),
    ),
    (
        "L2 · typical combination",
        result["L2_mahal"],
        (
            f"Squared Mahalanobis distance d² = {result['d2']:.2f} against a threshold of "
            f"{mahal['threshold_d2']:.2f}. Every value can be unremarkable on its own and "
            f"still form a combination no real wine shows."
        ),
    ),
    (
        "L3 · density consistent with alcohol and sugar",
        result["L3_density"],
        (
            f"Density {x['density']:.5f} against {density_pred:.5f} expected from this alcohol "
            f"and sugar: {result['z_density']:.1f} σ away, where the limit is "
            f"{density_band['k']:.0f} σ."
        ),
    ),
]
grid = st.columns(2)
for i, (name, ok, detail) in enumerate(layers):
    with grid[i % 2].container(border=True, height=190):
        head, status = st.columns([3, 1], vertical_alignment="center")
        head.markdown(f"**{name}**")
        with status:
            if ok:
                st.badge("Pass", icon=":material/check_circle:", color="green")
            else:
                st.badge("Fail", icon=":material/cancel:", color="red")
        st.caption(detail)

st.subheader("The physical law behind L3", icon=":material/water_drop:")
st.markdown(
    "Density is not a free parameter. Alcohol makes wine lighter and sugar makes it heavier, and "
    "the relationship is tight enough to fit from the data. Raise alcohol while holding density "
    "fixed and you have described a liquid that does not exist — which is precisely the kind of "
    "answer an unconstrained optimiser will hand you. Each grey dot is a real wine; the shaded "
    "band is the ±3σ window the search has to stay inside."
)


@st.cache_data(show_spinner=False)
def density_residuals(_wines: pd.DataFrame, coef: dict) -> pd.DataFrame:
    pred = (
        coef["const"]
        + coef["alcohol"] * _wines["alcohol"]
        + coef["residual sugar"] * _wines["residual sugar"]
    )
    return pd.DataFrame(
        {"alcohol": _wines["alcohol"], "density residual": _wines["density"] - pred}
    )


resid = density_residuals(wines, c)
band = density_band["k"] * density_band["sigma"]
this_wine = pd.DataFrame(
    {
        "alcohol": [x["alcohol"]],
        "density residual": [x["density"] - density_pred],
        "label": ["wine under test"],
    }
)
y_scale = alt.Scale(domain=[-3 * band, 3 * band], clamp=True)
with st.container(border=True):
    st.altair_chart(
        alt.layer(
            alt.Chart(pd.DataFrame({"lo": [-band], "hi": [band]}))
            .mark_rect(opacity=0.12, color=BAND)
            .encode(y=alt.Y("lo:Q", scale=y_scale), y2="hi:Q"),
            alt.Chart(resid)
            .mark_circle(size=14, opacity=0.3, color=CLOUD)
            .encode(
                x=alt.X(
                    "alcohol:Q", scale=alt.Scale(zero=False), title="Alcohol (% vol)"
                ),
                y=alt.Y(
                    "density residual:Q",
                    scale=y_scale,
                    title="Density − law prediction (g/cm³)",
                ),
                tooltip=[
                    alt.Tooltip("alcohol:Q", format=".1f"),
                    alt.Tooltip("density residual:Q", format=".5f"),
                ],
            ),
            alt.Chart(this_wine)
            .mark_point(
                shape="diamond",
                size=280,
                filled=True,
                color=SELECTED,
                stroke="white",
                strokeWidth=1.2,
            )
            .encode(
                x="alcohol:Q",
                y=alt.Y("density residual:Q", scale=y_scale),
                tooltip=["label:N", alt.Tooltip("density residual:Q", format=".5f")],
            ),
        ).properties(
            height=350, title="Where the wine under test sits against the density law"
        ),
        alt="Density residual versus alcohol for all wines, with the edited wine highlighted",
    )
    st.caption(
        f"Fitted law: density ≈ {c['const']:.5f} {c['alcohol']:+.5f}·alcohol "
        f"{c['residual sugar']:+.5f}·sugar, with σ = {density_band['sigma']:.5f}. The blue "
        f"diamond is the wine under test; points beyond the chart edge are clamped to it."
    )

provenance(
    "Layer definitions and fitted thresholds: src/modeling/constraints.py, with the artifacts "
    "loaded from the Phase 4 MLflow run. Legal caps: Commission Delegated Regulation (EU) "
    "2019/934."
)
