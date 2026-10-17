"""Phase 3: Box-Behnken design on the Phase 1 simulator, second-order response surfaces and the
desirability optimum computed in R (r/rsm_fit.Rmd, r/desirability_optimize.R)."""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import REPO_ROOT, download_csv, provenance  # noqa: E402
from src.process.doe import FACTOR_LEVELS, decode, run_single_point  # noqa: E402

DESIGN_CSV = REPO_ROOT / "results" / "phase3_doe" / "design_matrix.csv"
OPTIMUM_CSV = REPO_ROOT / "results" / "tables" / "optimal_conditions.csv"

FACTORS = {
    "x1": ("T", "Temperature (°C)"),
    "x2": ("S0", "Sugar S₀ (g/L)"),
    "x3": ("N0", "Nitrogen N₀ (mg/L)"),
}
RESPONSES = {
    "time": "Fermentation time (h)",
    "peak_rate": "Peak sugar-uptake rate (g/L/h)",
    "conversion": "Sugar conversion",
    "ethanol": "Final ethanol (g/L)",
}


def quad_terms(X: np.ndarray) -> np.ndarray:
    """Full second-order model in coded units: 1, x_i, x_i·x_j, x_i²."""
    cols = [np.ones(len(X))]
    cols += [X[:, i] for i in range(3)]
    cols += [X[:, i] * X[:, j] for i, j in itertools.combinations(range(3), 2)]
    cols += [X[:, i] ** 2 for i in range(3)]
    return np.column_stack(cols)


@st.cache_data(show_spinner=False)
def load_design():
    design = pd.read_csv(DESIGN_CSV).rename(columns={"Ti": "T", "Sub0": "S0"})
    optimum = pd.read_csv(OPTIMUM_CSV).iloc[0]
    X = quad_terms(design[["x1", "x2", "x3"]].to_numpy())
    fits = {}
    for r in RESPONSES:
        y = design[r].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        fits[r] = {
            "beta": beta,
            "r2": 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())),
        }
    return design, optimum, fits


@st.cache_data(show_spinner="Running the simulator...", max_entries=64)
def simulate(T: float, S0: float, N0: float) -> dict:
    return run_single_point(T, S0, N0)


def code(value: float, factor: str) -> float:
    lv = FACTOR_LEVELS[factor]
    return (value - lv["center"]) / (lv["center"] - lv["low"])


def predict(fits, response, coded_xyz) -> np.ndarray:
    return quad_terms(np.atleast_2d(coded_xyz)) @ fits[response]["beta"]


design, optimum, fits = load_design()

st.markdown(
    "Fifteen virtual experiments on the fermentation twin, arranged as a three-factor "
    "Box–Behnken design over temperature, sugar and nitrogen. Each response is fitted with a "
    "second-order polynomial, and a desirability function picks the best compromise between a "
    "short, gentle fermentation and full sugar conversion."
)

st.subheader("Optimum from the desirability search", icon=":material/target:")
with st.container(horizontal=True):
    st.metric("Temperature", f"{optimum['T_C']:.0f} °C", border=True)
    st.metric("Sugar S₀", f"{optimum['S0_gL']:.1f} g/L", border=True)
    st.metric("Nitrogen N₀", f"{optimum['N0_mgL']:.1f} mg/L", border=True)
    st.metric("Predicted time", f"{optimum['pred_time']:.1f} h", border=True)
    st.metric(
        "Overall desirability D",
        f"{optimum['overall_D']:.3f}",
        border=True,
        help="Geometric mean of the individual desirability scores. 1.0 would satisfy every "
        "target perfectly.",
    )

st.warning(
    "**This optimum is a direction, not a setpoint.** Temperature sits exactly at the top "
    "of the design range (coded +1), so what the design supports is *warmer is better inside this "
    "window*, not *27 °C is optimal*. The desirability function also carries no aroma term, and "
    "cooler fermentations are generally better for white-wine aroma retention.",
    icon=":material/warning:",
)

tab_surface, tab_check, tab_design = st.tabs(
    [
        ":material/grid_on: Response surface",
        ":material/science: Test a point",
        ":material/table_chart: Design and fits",
    ]
)

with tab_surface:
    c_in, c_plot = st.columns([1, 3])
    with c_in:
        response = st.selectbox("Response", list(RESPONSES), format_func=RESPONSES.get)
        held = st.selectbox(
            "Hold this factor fixed",
            list(FACTORS),
            index=1,
            format_func=lambda k: FACTORS[k][1],
        )
        held_name = FACTORS[held][0]
        lv = FACTOR_LEVELS[held_name]
        held_value = st.slider(
            FACTORS[held][1],
            float(lv["low"]),
            float(lv["high"]),
            float(decode(float(optimum[f"{held}_coded"]), held_name)),
        )
        st.metric(
            "Fit R² for this response",
            f"{fits[response]['r2']:.3f}",
            border=True,
            help="How much of the variation across the 15 design runs the quadratic captures.",
        )

    free = [k for k in FACTORS if k != held]
    grid = np.linspace(-1, 1, 41)
    g1, g2 = np.meshgrid(grid, grid)
    coded = np.zeros((g1.size, 3))
    idx = {k: i for i, k in enumerate(FACTORS)}
    coded[:, idx[free[0]]] = g1.ravel()
    coded[:, idx[free[1]]] = g2.ravel()
    coded[:, idx[held]] = code(held_value, held_name)
    surf = pd.DataFrame(
        {
            FACTORS[free[0]][1]: [decode(v, FACTORS[free[0]][0]) for v in g1.ravel()],
            FACTORS[free[1]][1]: [decode(v, FACTORS[free[1]][0]) for v in g2.ravel()],
            RESPONSES[response]: predict(fits, response, coded),
        }
    )
    xs, ys = FACTORS[free[0]][1], FACTORS[free[1]][1]
    step_x = (
        FACTOR_LEVELS[FACTORS[free[0]][0]]["high"]
        - FACTOR_LEVELS[FACTORS[free[0]][0]]["low"]
    ) / 40
    step_y = (
        FACTOR_LEVELS[FACTORS[free[1]][0]]["high"]
        - FACTOR_LEVELS[FACTORS[free[1]][0]]["low"]
    ) / 40
    surf["x2_"] = surf[xs] + step_x
    surf["y2_"] = surf[ys] + step_y

    pts = design.rename(columns={FACTORS[k][0]: FACTORS[k][1] for k in FACTORS})
    pts = pts[np.isclose(design[held_name], held_value)]
    opt_pt = pd.DataFrame(
        {
            xs: [decode(float(optimum[f"{free[0]}_coded"]), FACTORS[free[0]][0])],
            ys: [decode(float(optimum[f"{free[1]}_coded"]), FACTORS[free[1]][0])],
            "label": ["desirability optimum"],
        }
    )

    chart = (
        alt.Chart(surf)
        .mark_rect()
        .encode(
            x=alt.X(f"{xs}:Q", scale=alt.Scale(zero=False)),
            x2="x2_",
            y=alt.Y(f"{ys}:Q", scale=alt.Scale(zero=False)),
            y2="y2_",
            color=alt.Color(f"{RESPONSES[response]}:Q"),
            tooltip=[
                alt.Tooltip(xs, format=".1f"),
                alt.Tooltip(ys, format=".1f"),
                alt.Tooltip(RESPONSES[response], format=".3f"),
            ],
        )
    )
    layers = [chart]
    if len(pts):
        layers.append(
            alt.Chart(pts)
            .mark_point(
                size=90, filled=True, color="white", stroke="#1C1E21", strokeWidth=1.2
            )
            .encode(
                x=f"{xs}:Q",
                y=f"{ys}:Q",
                tooltip=[
                    alt.Tooltip(xs, format=".1f"),
                    alt.Tooltip(ys, format=".1f"),
                    alt.Tooltip(f"{response}:Q", format=".3f", title="measured"),
                ],
            )
        )
    layers.append(
        alt.Chart(opt_pt)
        .mark_point(
            shape="cross",
            size=260,
            filled=True,
            color="#D03B3B",
            stroke="white",
            strokeWidth=1,
        )
        .encode(x=f"{xs}:Q", y=f"{ys}:Q", tooltip=["label:N"])
    )
    with c_plot.container(border=True):
        st.altair_chart(
            alt.layer(*layers).properties(height=430, title=RESPONSES[response]),
            alt=f"Fitted response surface of {RESPONSES[response]}",
        )
        st.caption(
            "White circles are the design runs that sit in this slice; the red cross is the "
            "desirability optimum projected onto the plane. The surface is a quadratic fitted "
            "in Python with the same terms as the R model, so values match the R fit up to "
            "rounding."
        )

with tab_check:
    st.markdown(
        "The surfaces are an approximation of the simulator, and an approximation can be wrong. "
        "Pick conditions inside the design window: the full simulator runs, and its result is "
        "compared with what the fitted surface predicts. A large gap means the quadratic should "
        "not be trusted there."
    )
    with st.form("test_point", border=True):
        a, b, c = st.columns(3)
        T = a.slider("Temperature (°C)", 15.0, 27.0, float(optimum["T_C"]), 0.5)
        S0 = b.slider(
            "Sugar S₀ (g/L)", 250.0, 280.0, float(round(optimum["S0_gL"])), 1.0
        )
        N0 = c.slider(
            "Nitrogen N₀ (mg/L)", 140.0, 340.0, float(round(optimum["N0_mgL"])), 5.0
        )
        st.form_submit_button(
            "Run simulation", icon=":material/play_arrow:", type="primary"
        )

    sim = simulate(T, S0, N0)
    xyz = [code(T, "T"), code(S0, "S0"), code(N0, "N0")]
    comp = pd.DataFrame(
        [
            {
                "response": RESPONSES[r],
                "simulator": float(sim[r]),
                "surface": float(predict(fits, r, xyz)[0]),
            }
            for r in RESPONSES
        ]
    )
    comp["difference"] = comp["surface"] - comp["simulator"]
    st.dataframe(
        comp.round(3),
        hide_index=True,
        width="stretch",
        alt="Simulator versus fitted-surface predictions at the chosen point",
        column_config={
            "difference": st.column_config.NumberColumn(
                "difference (surface − simulator)", format="%.3f"
            )
        },
    )
    if not sim["reached_dryness"]:
        st.warning(
            "This point never got below 4 g/L of sugar: a stuck fermentation, usually from low "
            "nitrogen.",
            icon=":material/warning:",
        )
    else:
        st.success(
            "This point reached dryness in the simulator.",
            icon=":material/check_circle:",
        )

with tab_design:
    n_stuck = int((~design["reached_dryness"]).sum())
    st.markdown(
        f"{len(design)} runs, including three centre points. Because the simulator is "
        f"deterministic the centre replicates are identical — there is no pure-error estimate, "
        f"which is the price of experimenting on a model rather than in a cellar. "
        f"{n_stuck} run(s) did not reach 4 g/L of residual sugar."
    )
    table = design[
        ["x1", "x2", "x3", "T", "S0", "N0", *RESPONSES, "reached_dryness"]
    ].round(3)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        alt="Box-Behnken design matrix and results",
    )
    download_csv(
        table, "Design matrix (CSV)", "phase3_design_matrix.csv", key="dl_design"
    )

    st.markdown("**Quality of the quadratic fits**")
    r2 = pd.DataFrame(
        {
            "response": [RESPONSES[r] for r in RESPONSES],
            "R² of quadratic fit": [fits[r]["r2"] for r in RESPONSES],
        }
    ).round(4)
    st.dataframe(r2, hide_index=True, alt="R-squared of each fitted response surface")

provenance(
    "Design and simulator runs: src/process/doe.py. Response surfaces and the desirability "
    "optimum: r/rsm_fit.Rmd and r/desirability_optimize.R, with results in results/tables/."
)
