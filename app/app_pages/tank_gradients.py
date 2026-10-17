"""Phase 2: CFD-lite. 1D heat transport up the tank height (src/process/spatial.py), driven by the
Phase 1 sugar kinetics, then a check of whether the resulting gradient changes the fermentation.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import provenance  # noqa: E402
from src.process.simulator import params as ferm_params, run_simulation  # noqa: E402
from src.process.spatial import (
    params as spatial_params,
    run_with_fermentation,
)  # noqa: E402

ALPHA_OPTIONS = {
    "1.33e-7 (molecular, conduction only)": 1.33e-7,
    "1.0e-6": 1.0e-6,
    "1.0e-5": 1.0e-5,
    "1.9e-5 (model default)": 1.9e-5,
    "1.33e-4 (strong natural convection)": 1.33e-4,
}


@st.cache_data(show_spinner="Solving heat transport...", max_entries=32)
def solve_tank(T_coolant: float, alpha: float, H: float, N: int):
    """Return (t, z, T[z, t]) for the coupled heat + fermentation-source run."""
    p = copy.deepcopy(spatial_params)
    p.update(T_coolant=T_coolant, T_initial=T_coolant, alpha=alpha, H=H, N=N)
    p["dz"] = p["H"] / (p["N"] - 1)
    p["alpha_h"] = p["alpha"] * 3600
    sol, _ = run_with_fermentation(p, dict(ferm_params))
    return sol.t, np.linspace(0, H, N), sol.y


@st.cache_data(show_spinner=False, max_entries=64)
def ferment_at(T: float):
    """Phase 1 run at a fixed temperature; returns (t, sugar, ethanol)."""
    sol = run_simulation(dict(ferm_params, T=T), t_end=700)
    return sol.t, sol.y[2], sol.y[4]


def time_to_dry(t, S, threshold=1.0):
    below = np.where(S < threshold)[0]
    return float(t[below[0]]) if below.size else None


st.markdown(
    "The fermentation twin assumes one temperature for the whole tank. Here that assumption is "
    "tested: the tank is cut into horizontal layers and heat moves between them. Fermentation "
    "releases heat everywhere, the cooling jacket holds the base at the coolant temperature, and "
    "the open top loses almost nothing — so the tank stratifies into a warm top and a cold base. "
    "Set the vessel and the cooling in the sidebar; the test below is whether that gradient "
    "changes the wine."
)

with st.sidebar:
    st.subheader("Tank and cooling", icon=":material/tune:")
    T_coolant = st.slider("Coolant / base temperature (°C)", 8.0, 20.0, 12.0, 0.5)
    alpha_label = st.selectbox(
        "Effective thermal diffusivity α (m²/s)",
        list(ALPHA_OPTIONS),
        index=3,
        help="Molecular diffusivity alone gives almost no mixing in a fermentation's time. "
        "Larger values stand in for buoyancy-driven convection.",
    )
    H = st.slider("Tank height (m)", 1.0, 5.0, float(spatial_params["H"]), 0.1)
    N = st.segmented_control("Layers N", [10, 20, 40, 80], default=40) or 40
    st.caption(
        "Fermentation kinetics use the white-wine preset. Changing N should barely move the "
        "answer — that is the grid-independence check."
    )

t, z, Tz = solve_tank(T_coolant, ALPHA_OPTIONS[alpha_label], H, N)
gradient = Tz[-1] - Tz[0]
k_peak = int(np.argmax(gradient))
T_top_mean = float(Tz[-1].mean())

with st.container(horizontal=True):
    st.metric(
        "Peak top-to-base gradient",
        f"{gradient[k_peak]:.2f} °C",
        border=True,
        help="Largest difference between the top and base layers at any point in the run.",
    )
    st.metric("Peak reached at", f"{t[k_peak]:.0f} h", border=True)
    st.metric("Mean top temperature", f"{T_top_mean:.1f} °C", border=True)
    st.metric("Simulated length", f"{t[-1]:.0f} h", border=True)

st.subheader("Temperature through the tank", icon=":material/thermostat:")

# Heatmap of T(z, t), thinned in time to keep the chart light.
step = max(1, len(t) // 120)
ts = t[::step]
heat = pd.DataFrame(
    {
        "time (h)": np.repeat(ts, len(z)),
        "height (m)": np.tile(z, len(ts)),
        "T (°C)": Tz[:, ::step].T.ravel(),
    }
)
dt = float(ts[1] - ts[0]) if len(ts) > 1 else 1.0
dz = float(z[1] - z[0])
heat["t2"] = heat["time (h)"] + dt
heat["z2"] = heat["height (m)"] + dz
with st.container(border=True):
    st.altair_chart(
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("time (h):Q", title="Time (h)"),
            x2="t2",
            y=alt.Y("height (m):Q", title="Height above the base (m)"),
            y2="z2",
            color=alt.Color("T (°C):Q", title="T (°C)"),
            tooltip=[
                alt.Tooltip("time (h):Q", format=".0f"),
                alt.Tooltip("height (m):Q", format=".2f"),
                alt.Tooltip("T (°C):Q", format=".2f"),
            ],
        )
        .properties(height=300),
        alt="Heatmap of temperature by tank height and time",
    )
    st.caption(
        "Warm band at the top, cold band at the jacketed base. The gradient builds with the "
        "heat of fermentation and decays as the sugar runs out."
    )

left, right = st.columns(2)
picks = sorted({0, len(t) // 5, k_peak, 3 * len(t) // 4, len(t) - 1})
prof = pd.concat(
    pd.DataFrame(
        {"T (°C)": Tz[:, k], "height (m)": z, "time": f"{t[k]:.0f} h", "order": k}
    )
    for k in picks
)
with left.container(border=True):
    st.altair_chart(
        alt.Chart(prof)
        .mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2)
        .encode(
            x=alt.X("T (°C):Q", scale=alt.Scale(zero=False), title="T (°C)"),
            y=alt.Y("height (m):Q", title="Height above the base (m)"),
            color=alt.Color(
                "time:N",
                sort=alt.SortField("order"),
                title="Elapsed",
                legend=alt.Legend(orient="top"),
            ),
            order="height (m):Q",
            tooltip=[
                "time:N",
                alt.Tooltip("height (m):Q", format=".2f"),
                alt.Tooltip("T (°C):Q", format=".2f"),
            ],
        )
        .properties(height=280, title="Vertical profiles"),
        alt="Temperature profile up the tank at several times",
    )
with right.container(border=True):
    st.altair_chart(
        alt.Chart(pd.DataFrame({"time (h)": t, "top − base (°C)": gradient}))
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time (h):Q", title="Time (h)"),
            y=alt.Y("top − base (°C):Q", title="Top − base (°C)"),
            tooltip=[
                alt.Tooltip("time (h):Q", format=".0f"),
                alt.Tooltip("top − base (°C):Q", format=".2f"),
            ],
        )
        .properties(height=280, title="Top-to-base gradient"),
        alt="Top minus base temperature over time",
    )

st.subheader("Does the gradient change the answer?", icon=":material/compare_arrows:")
st.markdown(
    f"The test: run the fermentation twin twice, once at the base temperature "
    f"({T_coolant:.1f} °C) and once at the mean temperature the top layer actually sees "
    f"({T_top_mean:.1f} °C), and compare the two wines."
)

runs = {
    f"Cold base ({T_coolant:.1f} °C)": ferment_at(T_coolant),
    f"Warm top ({T_top_mean:.1f} °C)": ferment_at(round(T_top_mean, 2)),
}
frames, dry = [], {}
for label, (tt, S, E) in runs.items():
    frames.append(
        pd.DataFrame(
            {"time (h)": tt, "sugar (g/L)": S, "ethanol (g/L)": E, "layer": label}
        )
    )
    dry[label] = (time_to_dry(tt, S), float(E[-1]))
traj = pd.concat(frames)

(label_cold, (dry_cold, e_cold)), (label_warm, (dry_warm, e_warm)) = dry.items()
with st.container(horizontal=True):
    st.metric(
        "Base reaches dryness",
        f"{dry_cold:.0f} h" if dry_cold else "not within 700 h",
        border=True,
    )
    st.metric(
        "Top reaches dryness",
        f"{dry_warm:.0f} h" if dry_warm else "not within 700 h",
        delta=f"{dry_warm - dry_cold:+.0f} h" if dry_cold and dry_warm else None,
        delta_color="off",
        border=True,
        help="Difference against the base layer.",
    )
    st.metric(
        "Final ethanol, base / top",
        f"{e_cold:.1f} / {e_warm:.1f} g/L",
        border=True,
        help="The two layers converge on the same ethanol. Only the timing differs.",
    )

c1, c2 = st.columns(2)
for col, y, label in (
    (c1, "sugar (g/L)", "Sugar (g/L)"),
    (c2, "ethanol (g/L)", "Ethanol (g/L)"),
):
    with col.container(border=True):
        st.altair_chart(
            alt.Chart(traj)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("time (h):Q", title="Time (h)"),
                y=alt.Y(f"{y}:Q", title=label),
                color=alt.Color("layer:N", title=None, legend=alt.Legend(orient="top")),
                tooltip=[
                    "layer:N",
                    alt.Tooltip("time (h):Q", format=".0f"),
                    alt.Tooltip(f"{y}:Q", format=".2f", title=label),
                ],
            )
            .properties(height=280, title=label),
            alt=f"{label} over time at the base and top temperatures",
        )

st.caption(
    "The gradient shifts the timing, not the destination. Both layers end at "
    "about the same ethanol, but the warm top finishes earlier. This is 1D conduction with an "
    "effective diffusivity, not full CFD — it establishes whether the gradient matters, it does "
    "not resolve the flow field."
)

provenance(
    "Heat transport: src/process/spatial.py, coupled to the Phase 1 kinetics as a heat source. "
    "Grid independence is checked by varying N in the sidebar."
)
