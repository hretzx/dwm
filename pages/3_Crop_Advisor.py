"""
pages/3_Crop_Advisor.py
-----------------------
Crop Advisor: predicts a crop with our own ID3 decision tree, shows the exact
decision path used, and lists 3 alternative crops by profile fit.
"""
import numpy as np
import pandas as pd
import streamlit as st

from utils.models import train_id3
from utils.preprocessing import TARGET
from utils.session import FEATURES, require_assessment
from utils.ui import PRIMARY, page_header

LIGHT = "#E8F5E9"
PRETTY = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)", "ph": "pH",
          "temperature": "Temperature (°C)", "humidity": "Humidity (%)", "rainfall": "Rainfall (mm)"}


def profile_fit_ranking(df: pd.DataFrame, x: dict, exclude: str) -> pd.DataFrame:
    """
    Rank crops by how close the input is to each crop's AVERAGE profile.
    For every feature we measure the distance from the crop's mean in units of
    that crop's standard deviation (a z-distance); the average over all
    features is turned into a 0-100 fit score. Lower distance = higher score.
    """
    overall_std = df[FEATURES].std()
    rows = []
    for crop, g in df.groupby(TARGET):
        mean, std = g[FEATURES].mean(), g[FEATURES].std()
        std = np.maximum(std, 0.1 * overall_std)           # avoid divide-by-tiny
        z = ((pd.Series(x) - mean).abs() / std).mean()
        rows.append({"Crop": crop, "Profile fit": round(100 / (1 + z), 1)})
    out = pd.DataFrame(rows).sort_values("Profile fit", ascending=False)
    return out[out["Crop"] != exclude].head(3).reset_index(drop=True)


# ------------------------------------------------------------------- the page
page_header(
    "Crop Advisor",
    "A decision tree (ID3) built from scratch predicts the crop that best matches "
    "your soil and weather values.",
)
st.caption(
    "ID3 first sorts each value into bins (for example 'rainfall: high'). It then "
    "asks the question that reduces uncertainty the most (information gain), "
    "step by step, until it reaches a crop."
)

x = require_assessment()
model, accuracy, df = train_id3()
crop, path, purity = model.predict_with_path(x)

# Remember the prediction for other pages
st.session_state["id3_prediction"] = crop

# --- Prediction card
st.markdown(
    f"""
    <div style="background:{LIGHT}; border-left:6px solid {PRIMARY}; border-radius:12px;
                padding:1.4rem 1.6rem; margin:0.5rem 0 1rem 0;">
        <div style="color:#555; font-size:0.9rem;">ID3 predicted crop</div>
        <div style="color:{PRIMARY}; font-size:2.4rem; font-weight:700;
                    text-transform:capitalize;">{crop}</div>
        <div style="color:#555; font-size:0.9rem;">
            In training data, {purity:.0%} of rows at the final tree node were this crop.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Decision path
st.subheader("Decision path for your input")
st.caption("Each row is one question the tree asked about your values, from the root downward.")
steps = pd.DataFrame([{
    "Step": i + 1,
    "Feature tested": PRETTY[s["feature"]],
    "Your value": round(s["value"], 2),
    "Falls in": s["level"],
    "Range of that bin": s["range"],
    "Info gain": round(s["gain"], 3),
    "Training rows here": s["rows_here"],
} for i, s in enumerate(path)])
if steps.empty:
    st.write("The tree predicted directly from the root without any questions.")
else:
    st.dataframe(steps, hide_index=True, width="stretch")
st.markdown(f"**Result:** reached the leaf for **{crop}**.")

# --- Alternatives
st.subheader("Alternative crops (profile fit)")
st.caption(
    "This is a separate ranking from the ID3 prediction. It compares your values with "
    "each crop's average profile in the dataset. A higher score means your input sits "
    "closer to that crop's typical values."
)
alts = profile_fit_ranking(df, x, exclude=crop)
cols = st.columns(3)
for col, (_, r) in zip(cols, alts.iterrows()):
    col.markdown(
        f"""
        <div style="background:white; border:1px solid {LIGHT}; border-radius:12px;
                    padding:1rem 1.2rem; box-shadow:0 1px 4px rgba(0,0,0,0.06);">
            <div style="color:#333; font-size:1.3rem; font-weight:600;
                        text-transform:capitalize;">{r['Crop']}</div>
            <div style="color:{PRIMARY}; font-size:1.6rem; font-weight:700;">{r['Profile fit']}</div>
            <div style="color:#777; font-size:0.8rem;">profile fit (0 to 100)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
with st.expander("Model details"):
    st.write(
        f"Held-out accuracy on 20% of the dataset: **{accuracy:.1%}**. "
        f"Continuous features were split into {model.n_bins} equal-frequency bins "
        f"and the tree depth is limited to {model.max_depth}."
    )
    edges = pd.DataFrame({PRETTY[f]: [", ".join(f"{e:.4g}" for e in model.edges[f])] for f in FEATURES},
                         index=["Bin edges"]).T
    st.dataframe(edges, width="stretch")
