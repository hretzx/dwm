"""
app.py
------
Entry point:  streamlit run app.py
 - sets the page config and theme
 - builds the sidebar navigation (only pages whose file exists are listed,
   so the app keeps working while pages are added one at a time)
 - contains the Dashboard page itself
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from db.warehouse import get_recent_assessments, init_db
from utils.preprocessing import DEFAULT_K, TARGET, load_dataset
from utils.ui import ACCENT, PRIMARY, apply_theme, page_header

st.set_page_config(page_title="AgriMine", layout="wide")
ROOT = Path(__file__).parent


@st.cache_data
def get_data() -> tuple[pd.DataFrame, bool]:
    """Load the dataset once and reuse it on every rerun."""
    return load_dataset()


# ----------------------------------------------------------------- Dashboard
def dashboard() -> None:
    df, is_synthetic = get_data()
    init_db()

    page_header(
        "Dashboard",
        "Overview of the crop dataset and your saved soil assessments. "
        "All numbers are calculated from the loaded data.",
    )
    if is_synthetic:
        st.warning(
            "Using a generated sample dataset. Place the Kaggle file at "
            "data/Crop_recommendation.csv and restart to use real data."
        )

    # --- Metric cards
    # The cluster count is the k chosen on the Cluster Explorer page
    # (defaults to DEFAULT_K until that page sets it).
    k = st.session_state.get("kmeans_k", DEFAULT_K)
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Crop classes", df[TARGET].nunique())
    c3.metric("Clusters (K-Means)", k, help="Set on the Cluster Explorer page.")

    st.write("")

    # --- Crop distribution
    st.subheader("Crop distribution")
    counts = (df[TARGET].value_counts().rename_axis("Crop").reset_index(name="Records"))
    fig = px.bar(counts, x="Crop", y="Records", color_discrete_sequence=[PRIMARY])
    fig.update_traces(marker_line_width=0, hovertemplate="%{x}: %{y} records<extra></extra>")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", height=340,
        margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, xaxis_tickangle=-40,
    )
    fig.update_yaxes(gridcolor="#E8F5E9")
    st.plotly_chart(fig, width="stretch")

    # --- New assessment button
    st.write("")
    if st.button("New Soil Assessment", type="primary"):
        target = ROOT / "pages" / "1_Soil_Analysis.py"
        if target.exists():
            st.switch_page(str(target))
        else:
            st.info("The Soil Analysis page has not been built yet. It is the next step.")

    # --- Recent assessments
    st.subheader("Recent assessments")
    recent = get_recent_assessments(10)
    if recent.empty:
        st.caption("No assessments saved yet. Start with a new soil assessment.")
    else:
        recent = recent.rename(columns={
            "id": "ID", "created_at": "Saved at", "ph": "pH",
            "temperature": "Temp (°C)", "humidity": "Humidity (%)",
            "rainfall": "Rainfall (mm)", "predicted_crop": "Predicted crop",
            "cluster_id": "Cluster",
        })
        st.dataframe(recent, hide_index=True, width="stretch")


# ---------------------------------------------------------------- Navigation
apply_theme()

pages = [st.Page(dashboard, title="Dashboard", default=True, url_path="dashboard")]

# Page files are named like "1_Soil_Analysis.py" -> title "Soil Analysis".
for f in sorted((ROOT / "pages").glob("*.py")):
    title = f.stem.split("_", 1)[1].replace("_", " ")
    pages.append(st.Page(str(f), title=title))

st.navigation(pages).run()
