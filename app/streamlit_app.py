"""Wine Innovation Engine: interactive front end.

Run from the repo root:
    streamlit run app/streamlit_app.py

This module is the shell only. It sets the page config, declares the navigation, and renders
the page heading, so every page gets the same framing. Page bodies live in app/app_pages/ and
must not call st.title themselves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `common` importable

import streamlit as st  # noqa: E402

from common import APP_NAME, LOGO, MARK, TAGLINE  # noqa: E402

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":material/wine_bar:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "about": (
            f"**{APP_NAME}** — {TAGLINE}.\n\n"
            "Six linked models: a fermentation digital twin, a 1D thermal check on the "
            "well-mixed assumption, a Box–Behnken design and response-surface optimisation, "
            "a sensory-quality model with SHAP attributions, a four-layer feasibility filter, "
            "and a constrained reformulation search.\n\n"
            "Results are directional. The quality model explains about 40 % of the variance "
            "in human sensory scores."
        )
    },
)

st.logo(str(LOGO), icon_image=str(MARK), size="large")

page = st.navigation(
    {
        "": [
            st.Page(
                "app_pages/overview.py",
                title="Overview",
                icon=":material/dashboard:",
                url_path="overview",
                default=True,
            )
        ],
        "Process engineering": [
            st.Page(
                "app_pages/fermentation_twin.py",
                title="Fermentation twin",
                icon=":material/science:",
                url_path="fermentation-twin",
            ),
            st.Page(
                "app_pages/tank_gradients.py",
                title="Tank gradients",
                icon=":material/thermostat:",
                url_path="tank-gradients",
            ),
            st.Page(
                "app_pages/process_optimisation.py",
                title="Process optimisation",
                icon=":material/tune:",
                url_path="process-optimisation",
            ),
        ],
        "Product design": [
            st.Page(
                "app_pages/quality_model.py",
                title="Quality model",
                icon=":material/analytics:",
                url_path="quality-model",
            ),
            st.Page(
                "app_pages/feasibility.py",
                title="Feasibility",
                icon=":material/fact_check:",
                url_path="feasibility",
            ),
            st.Page(
                "app_pages/reformulation.py",
                title="Reformulation",
                icon=":material/auto_fix_high:",
                url_path="reformulation",
            ),
        ],
    },
    position="sidebar",
    expanded=True,
)

st.title(page.title, icon=page.icon)
page.run()

with st.sidebar:
    st.space("medium")
    st.caption(f"{APP_NAME} · {TAGLINE}")
    st.caption(
        "Quality model R² ≈ 0.40 on held-out wines. Predicted scores are guidance, not promises."
    )
