"""Phase 6: reformulation. Shows the saved optimiser runs behind the README's Phase 6 results:
a nearby wine with higher predicted quality, inside the Phase 5 feasibility layers, under two
scenarios."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (  # noqa: E402
    download_csv,
    list_searched_wines,
    load_phase6,
    load_saved_reformulation,
    provenance,
)

SCENARIO_LABELS = {
    "vintage_design": "Vintage design",
    "cellar_intervention": "Cellar intervention",
}
SCENARIO_SUBTITLES = {
    "vintage_design": "Every variable may move — decisions available at harvest and during "
    "fermentation as well as in the cellar.",
    "cellar_intervention": "Only additives may move — what can still be changed once the wine "
    "is made.",
}
SCENARIO_ICONS = {
    "vintage_design": ":material/agriculture:",
    "cellar_intervention": ":material/science:",
}
CHECK_LABELS = {
    "L0_legal": "L0 legal (SO₂ cap, volatile acidity)",
    "L1_bounds": "L1 within observed ranges",
    "L2_mahal": "L2 typical combination of values",
    "L3_density": "L3 density consistent with alcohol and sugar",
}

st.markdown(
    "Pick a real wine. The optimiser searches for a chemistry that scores higher on the quality "
    "model while staying legal, typical and physically consistent, and without drifting far from "
    "the wine it started from. It runs twice, under two commercially different sets of freedom."
)

model, artifacts, feature_order, wines = load_phase6()

searched = list_searched_wines()
if not searched:
    st.info(
        "No reformulation searches have been saved yet. Run "
        "`venv/bin/python scripts/run_phase6_batch.py` to search a sample of wines.",
        icon=":material/info:",
    )
    st.stop()


def _label(i: int) -> str:
    return f"Wine #{i} · panel score {int(wines.iloc[i]['quality'])}"


with st.sidebar:
    st.subheader("Wine to reformulate", icon=":material/tune:")
    wine_index = st.selectbox(
        f"Wine ({len(searched)} searched)",
        searched,
        index=searched.index(536) if 536 in searched else 0,
        format_func=_label,
    )
    st.caption(
        "Only wines with both scenarios already searched are listed, so a result is never "
        "shown half-finished."
    )

x0 = wines.iloc[wine_index][feature_order].values.astype(float)
q_now = float(model.predict([x0])[0])

saved = {s: load_saved_reformulation(wine_index, s) for s in SCENARIO_LABELS}

if not any(saved.values()):
    st.info(
        f"No saved reformulation exists for wine #{wine_index}. Choose another wine from the "
        "sidebar, or run `scripts/run_phase6_batch.py` to search more wines.",
        icon=":material/info:",
    )
    st.stop()

st.subheader(f"Starting wine #{wine_index}", icon=":material/wine_bar:")
with st.container(horizontal=True):
    st.metric("Predicted quality today", f"{q_now:.2f}", border=True)
    st.metric(
        "Panel score on record",
        f"{int(wines.iloc[wine_index]['quality'])}",
        border=True,
    )
    for scenario, label in SCENARIO_LABELS.items():
        r = saved[scenario]
        st.metric(
            f"{label} ceiling",
            f"{r['metrics']['quality_final']:.2f}" if r else "—",
            f"{r['metrics']['quality_gain']:+.2f}" if r else None,
            border=True,
            help=SCENARIO_SUBTITLES[scenario],
        )

with st.expander("Chemistry of the starting wine", icon=":material/table_chart:"):
    start = pd.DataFrame({"measurement": feature_order, "value": x0})
    st.dataframe(
        start.round(4), hide_index=True, width="stretch", alt="Starting wine chemistry"
    )

st.subheader("The two scenarios side by side", icon=":material/compare_arrows:")
cols = st.columns(2)
for col, (scenario, label) in zip(cols, SCENARIO_LABELS.items()):
    r = saved[scenario]
    with col.container(border=True):
        st.markdown(f"#### {SCENARIO_ICONS[scenario]} {label}")
        st.caption(SCENARIO_SUBTITLES[scenario])
        if r is None:
            st.warning("No saved run for this scenario.", icon=":material/warning:")
            continue
        m = r["metrics"]
        a, b = st.columns(2)
        a.metric(
            "Predicted quality",
            f"{m['quality_final']:.2f}",
            f"{m['quality_gain']:+.2f}",
            border=True,
            delta_description="vs. the starting wine",
        )
        b.metric(
            "Distance moved",
            f"{m['proximity_cost']:.2f}",
            border=True,
            help="How far the reformulated chemistry sits from the original, in scaled units. "
            "The search is penalised for drifting.",
        )

        st.markdown("**Feasibility of the proposed wine**")
        n_ok = sum(bool(r["audit"][k]) for k in CHECK_LABELS)
        if n_ok == len(CHECK_LABELS):
            st.success("All four layers pass.", icon=":material/verified:")
        else:
            st.error(
                f"{len(CHECK_LABELS) - n_ok} layer(s) fail.", icon=":material/block:"
            )
        for key, text in CHECK_LABELS.items():
            icon = ":material/check_circle:" if r["audit"][key] else ":material/cancel:"
            colour = ":green" if r["audit"][key] else ":red"
            st.markdown(f"{icon} {colour}[{text}]")

        st.markdown("**Largest proposed changes**")
        delta = r["delta"].rename(columns={"index": "feature"}).head(6)
        st.dataframe(
            delta[["feature", "before", "after", "delta"]].round(3),
            hide_index=True,
            width="stretch",
            alt=f"Largest changes proposed under the {label} scenario",
        )
        download_csv(
            r["delta"].rename(columns={"index": "feature"}).round(4),
            "All changes (CSV)",
            f"reformulation_wine{wine_index}_{scenario}.csv",
            key=f"dl_{scenario}",
        )

if saved["vintage_design"] and saved["cellar_intervention"]:
    gain_v = saved["vintage_design"]["metrics"]["quality_gain"]
    gain_c = saved["cellar_intervention"]["metrics"]["quality_gain"]
    ladder = pd.DataFrame(
        {
            "stage": [
                "Wine as made",
                f"{SCENARIO_LABELS['cellar_intervention']} ceiling",
                f"{SCENARIO_LABELS['vintage_design']} ceiling",
            ],
            "predicted quality": [q_now, q_now + gain_c, q_now + gain_v],
        }
    )
    st.subheader("Where the headroom actually sits", icon=":material/insights:")
    chart_col, text_col = st.columns([3, 2])
    with chart_col.container(border=True):
        st.altair_chart(
            alt.Chart(ladder)
            .mark_bar(cornerRadiusEnd=4, height=34)
            .encode(
                x=alt.X(
                    "predicted quality:Q",
                    scale=alt.Scale(zero=False, nice=True),
                    title="Predicted quality (sensory points)",
                ),
                y=alt.Y("stage:N", sort=ladder["stage"].tolist(), title=None),
                tooltip=[
                    "stage:N",
                    alt.Tooltip("predicted quality:Q", format=".2f"),
                ],
            )
            .properties(
                height=200, title=f"Wine #{wine_index}: reachable predicted quality"
            ),
            alt="Predicted quality of the wine as made and the ceiling under each scenario",
        )
    with text_col.container(border=True):
        st.markdown(
            f"""
**The gap is the finding.** Letting every variable move gains **{gain_v:+.2f}** points.
Restricting changes to additives gains **{gain_c:+.2f}**.

Most of the difference is alcohol and density — properties set at harvest and during
fermentation, not in the cellar. Commercially that says the leverage on this wine is a
vineyard and fermentation decision; by the time it reaches the cellar, most of the available
quality has already been fixed.
            """
        )

st.caption(
    "These are saved results from MLflow experiment `phase6_reformulation`, the same runs the "
    "README quotes, and they are directional: the quality model is trained on correlations "
    "between chemistry and score, so a predicted gain assumes those relationships still hold "
    "when the chemistry is deliberately moved."
)

provenance(
    "Optimiser: src/modeling/reformulate.py, constrained by the Phase 5 feasibility layers. "
    "Searches are run by scripts/run_phase6_batch.py and tracked in MLflow."
)
