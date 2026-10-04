"""Phase 1: fermentation digital twin, driven by the ODE simulator in src/process/simulator.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import download_csv, provenance  # noqa: E402
from src.process.simulator import params, run_simulation  # noqa: E402

st.markdown(
    "A five-equation kinetic model of a white-wine fermentation: viable and total biomass, sugar, "
    "assimilable nitrogen and ethanol, integrated as a system of ODEs. Change the starting "
    "conditions in the sidebar and the simulation reruns."
)


@st.cache_data(show_spinner="Integrating the fermentation...", max_entries=64)
def simulate(T: float, S0: float, N0: float) -> pd.DataFrame:
    """Run the twin and return the trajectory as a tidy frame."""
    sol = run_simulation(dict(params, T=T, S0=S0, N0=N0))
    X, Xt, S, N, E = sol.y
    return pd.DataFrame(
        {
            "time (h)": sol.t,
            "viable biomass (g/L)": X,
            "total biomass (g/L)": Xt,
            "sugar (g/L)": S,
            "nitrogen (mg/L)": N,
            "ethanol (g/L)": E,
        }
    )


with st.sidebar:
    st.subheader("Starting conditions", icon=":material/tune:")
    T = st.slider("Temperature (°C)", 10.0, 35.0, float(params["T"]), 0.5)
    S0 = st.slider("Initial sugar S₀ (g/L)", 150.0, 300.0, float(params["S0"]), 5.0)
    N0 = st.slider("Initial nitrogen N₀ (mg/L)", 50.0, 400.0, float(params["N0"]), 10.0)
    st.caption(
        "Defaults are the white-wine preset from the simulator. Below roughly 80 mg/L of "
        "nitrogen the ferment tends to stall."
    )

traj = simulate(T, S0, N0)
t = traj["time (h)"].to_numpy()

# Time when assimilable nitrogen is effectively used up (first sample below 1 mg/L).
n_exhausted = np.where(traj["nitrogen (mg/L)"].to_numpy() < 1.0)[0]
t_n_out = float(t[n_exhausted[0]]) if n_exhausted.size else None
# Sugar approaches zero asymptotically, so the simulator's zero-crossing event does not fire
# reliably; judge completion from the final sugar level instead.
sugar_left = float(traj["sugar (g/L)"].iloc[-1])
finished = sugar_left < 1.0
sugar_at_n_out = (
    float(traj["sugar (g/L)"].iloc[n_exhausted[0]]) if n_exhausted.size else None
)

with st.container(horizontal=True):
    st.metric(
        "Nitrogen exhausted at",
        f"{t_n_out:.0f} h" if t_n_out else "not within run",
        border=True,
        help="First time assimilable nitrogen falls below 1 mg/L.",
    )
    st.metric(
        "Sugar still present then",
        f"{sugar_at_n_out:.0f} g/L" if sugar_at_n_out is not None else "—",
        border=True,
        help="Sugar remaining at the moment nitrogen runs out. This is the point of the page.",
    )
    st.metric("Final ethanol", f"{traj['ethanol (g/L)'].iloc[-1]:.1f} g/L", border=True)
    st.metric("Residual sugar at end", f"{sugar_left:.1f} g/L", border=True)
    st.metric(
        "Peak viable biomass",
        f"{traj['viable biomass (g/L)'].max():.2f} g/L",
        border=True,
    )

if finished:
    st.success(
        "Sugar was consumed: the fermentation ran to dryness under these conditions.",
        icon=":material/check_circle:",
    )
else:
    st.warning(
        f"{sugar_left:.0f} g/L of sugar was left at the end of the simulated window. This is a "
        "stuck or incomplete fermentation.",
        icon=":material/warning:",
    )


def trace(y: str, title: str, series: list[str] | None = None, height: int = 300):
    """One trace panel. A single series is named by the title, so it carries no legend."""
    cols = series or [y]
    long = traj.melt("time (h)", cols, var_name="series", value_name="value")
    enc = {
        "x": alt.X("time (h):Q", title="Time (h)"),
        "y": alt.Y("value:Q", title=title, scale=alt.Scale(zero=False)),
        "tooltip": [
            alt.Tooltip("time (h):Q", format=".0f"),
            alt.Tooltip("series:N", title=None),
            alt.Tooltip("value:Q", format=".2f", title=title),
        ],
    }
    if len(cols) > 1:
        enc["color"] = alt.Color(
            "series:N", title=None, legend=alt.Legend(orient="top")
        )
    layers = [alt.Chart(long).mark_line(strokeWidth=2).encode(**enc)]
    if t_n_out is not None:
        marker = pd.DataFrame({"time (h)": [t_n_out], "label": ["nitrogen out"]})
        layers.append(
            alt.Chart(marker)
            .mark_rule(strokeDash=[4, 3], strokeWidth=1.5, opacity=0.7)
            .encode(x="time (h):Q", tooltip=[alt.Tooltip("time (h):Q", format=".0f")])
        )
    return alt.layer(*layers).properties(height=height, title=title)


st.subheader("Trajectories", icon=":material/show_chart:")
row1 = st.columns(2)
row1[0].altair_chart(
    trace("sugar (g/L)", "Sugar (g/L)"), alt="Sugar concentration over time"
)
row1[1].altair_chart(
    trace("ethanol (g/L)", "Ethanol (g/L)"), alt="Ethanol concentration over time"
)
row2 = st.columns(2)
row2[0].altair_chart(
    trace("nitrogen (mg/L)", "Assimilable nitrogen (mg/L)"),
    alt="Assimilable nitrogen over time",
)
row2[1].altair_chart(
    trace(
        "viable biomass (g/L)",
        "Biomass (g/L)",
        ["viable biomass (g/L)", "total biomass (g/L)"],
    ),
    alt="Viable and total biomass over time",
)

st.caption(
    "The dashed vertical line marks the moment nitrogen runs out. Sugar is typically still high "
    "at that point, and viable biomass stops growing while total biomass plateaus: nitrogen, not "
    "sugar, is the limiting nutrient."
)

with st.expander("Trajectory data", icon=":material/table_chart:"):
    st.dataframe(
        traj.round(3),
        hide_index=True,
        width="stretch",
        height=280,
        alt="Simulated trajectory",
    )
    download_csv(
        traj.round(4),
        "Trajectory (CSV)",
        f"fermentation_T{T:g}_S{S0:g}_N{N0:g}.csv",
        key="dl_traj",
    )

provenance(
    "Model and parameters: src/process/simulator.py. Solved with SciPy's stiff ODE integrator; "
    "the preset is the repository's white-wine baseline."
)
