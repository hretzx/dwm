"""
pages/4_What-if_Lab.py
----------------------
What-if Lab: pick a preferred crop, tweak the soil values with sliders and
compare BEFORE (saved assessment) vs AFTER (slider values) using the same
ID3 tree and K-Means model as the other pages.
"""
import math

import pandas as pd
import streamlit as st

from utils.models import train_id3, train_kmeans
from utils.preprocessing import DEFAULT_K, TARGET
from utils.session import FEATURES, require_assessment
from utils.ui import PRIMARY, page_header

LIGHT = "#E8F5E9"
PRETTY = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)", "ph": "pH",
          "temperature": "Temperature (°C)", "humidity": "Humidity (%)", "rainfall": "Rainfall (mm)"}
INT_FEATURES = {"N", "P", "K"}
STEP = {"ph": 0.1, "temperature": 0.5, "humidity": 1.0, "rainfall": 1.0}


def status(value: float, low: float, high: float) -> str:
    """Compare a value with a crop's typical range."""
    if value < low:
        return "Below range"
    if value > high:
        return "Above range"
    return "Within range"


def result_card(title: str, crop: str, cluster: int, mismatches: int, preferred: str) -> str:
    """HTML for one side (before / after) of the comparison."""
    match = crop == preferred
    return f"""
    <div style="background:{LIGHT if match else 'white'}; border:1px solid {LIGHT};
                border-radius:12px; padding:1.1rem 1.3rem; box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="color:#777; font-size:0.85rem; text-transform:uppercase;
                    letter-spacing:0.05em;">{title}</div>
        <div style="margin-top:0.6rem; color:#555; font-size:0.85rem;">ID3 prediction</div>
        <div style="color:{PRIMARY}; font-size:1.7rem; font-weight:700;
                    text-transform:capitalize;">{crop}</div>
        <div style="margin-top:0.6rem; color:#555; font-size:0.85rem;">K-Means cluster</div>
        <div style="color:#333; font-size:1.3rem; font-weight:600;">Cluster {cluster}</div>
        <div style="margin-top:0.6rem; color:#555; font-size:0.85rem;">Parameters outside {preferred}'s typical range</div>
        <div style="color:#333; font-size:1.3rem; font-weight:600;">{mismatches} of {len(FEATURES)}</div>
    </div>
    """


# ------------------------------------------------------------------- the page
page_header(
    "What-if Lab",
    "Change soil and weather values and see how the models react compared with "
    "the saved assessment.",
)
st.caption(
    "This page re-runs the ID3 tree and the K-Means model on your adjusted values. "
    "It helps you see which inputs push the result toward or away from a preferred crop."
)

baseline = require_assessment()
id3, _, df = train_id3()
k = st.session_state.get("kmeans_k", DEFAULT_K)
kmeans = train_kmeans(k)

# --- Slider setup: (re)start from the current assessment whenever it changes
bounds = {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES}
for f in FEATURES:  # widen bounds so the saved value always fits on its slider
    bounds[f] = (min(bounds[f][0], baseline[f]), max(bounds[f][1], baseline[f]))


def reset_sliders() -> None:
    """Put every slider back to the saved assessment value."""
    for f in FEATURES:
        st.session_state[f"wi_{f}"] = int(round(baseline[f])) if f in INT_FEATURES else float(baseline[f])


signature = tuple(baseline[f] for f in FEATURES)
if st.session_state.get("wi_signature") != signature:
    reset_sliders()
    st.session_state["wi_signature"] = signature

# --- Controls
crops = sorted(df[TARGET].unique())
default_crop = st.session_state.get("id3_prediction", crops[0])
preferred = st.selectbox("Preferred crop", crops,
                         index=crops.index(default_crop) if default_crop in crops else 0)

st.markdown("**Adjust values**")
adjusted = {}
left, right = st.columns(2)
for i, f in enumerate(FEATURES):
    lo, hi = bounds[f]
    with (left if i % 2 == 0 else right):
        if f in INT_FEATURES:
            adjusted[f] = float(st.slider(PRETTY[f], math.floor(lo), math.ceil(hi), key=f"wi_{f}"))
        else:
            adjusted[f] = st.slider(PRETTY[f], lo, hi, step=STEP[f], key=f"wi_{f}")

st.button("Reset to current assessment", on_click=reset_sliders)

# --- Typical range of the preferred crop: 10th to 90th percentile of its rows
crop_rows = df[df[TARGET] == preferred]
typical = {f: (float(crop_rows[f].quantile(0.10)), float(crop_rows[f].quantile(0.90))) for f in FEATURES}

# --- Run both models on BEFORE and AFTER
def evaluate(values: dict) -> tuple[str, int, int]:
    crop, _, _ = id3.predict_with_path(values)
    cluster = int(kmeans.predict([[values[f] for f in FEATURES]])[0])
    miss = sum(status(values[f], *typical[f]) != "Within range" for f in FEATURES)
    return crop, cluster, miss


b_crop, b_cluster, b_miss = evaluate(baseline)
a_crop, a_cluster, a_miss = evaluate(adjusted)

# --- Before vs after
st.subheader("Before vs after")
c1, c2 = st.columns(2)
c1.markdown(result_card("Before (saved assessment)", b_crop, b_cluster, b_miss, preferred),
            unsafe_allow_html=True)
c2.markdown(result_card("After (your changes)", a_crop, a_cluster, a_miss, preferred),
            unsafe_allow_html=True)

st.write("")
if a_crop == preferred:
    st.success(f"With these values, ID3 predicts {preferred}.")
else:
    st.info(f"With these values, ID3 predicts {a_crop}, not {preferred}.")

# --- Mismatch table
st.subheader(f"Parameters vs {preferred}'s typical range")
st.caption("Typical range = the middle 80% (10th to 90th percentile) of that crop's records in the dataset.")
table = pd.DataFrame([{
    "Parameter": PRETTY[f],
    "Typical range": f"{typical[f][0]:.1f} to {typical[f][1]:.1f}",
    "Before": round(baseline[f], 2),
    "Before status": status(baseline[f], *typical[f]),
    "After": round(adjusted[f], 2),
    "After status": status(adjusted[f], *typical[f]),
} for f in FEATURES])
st.dataframe(table, hide_index=True, width="stretch")

still = [PRETTY[f] for f in FEATURES if status(adjusted[f], *typical[f]) != "Within range"]
if still:
    st.markdown("**Still mismatched:** " + ", ".join(still))
else:
    st.markdown("**All parameters are now within the typical range for this crop.**")

st.caption("This simulates model behaviour and does not guarantee real soil outcomes.")
