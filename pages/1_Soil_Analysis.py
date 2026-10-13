"""
pages/1_Soil_Analysis.py
------------------------
Collects one soil/weather profile, validates it, labels each value as
low / medium / high, saves it to the database and stores it in
st.session_state["assessment"] so the other pages can use it.
"""
import streamlit as st

from db.warehouse import save_assessment
from utils.preprocessing import (FEATURES, VALID_RANGES, categorize,
                                 category_cutoffs, load_dataset, validate_inputs)
from utils.ui import page_header

page_header(
    "Soil Analysis",
    "Enter a soil and weather profile. Each value is compared with the whole dataset "
    "and labelled low, medium or high (bottom third, middle third, top third of the records). "
    "These labels describe where a value sits in the data; they are not a judgement of the soil.",
)


@st.cache_data
def get_cutoffs_and_medians():
    """Cut-offs for the categories and median values used as form defaults."""
    df, _ = load_dataset()
    return category_cutoffs(df), df[FEATURES].median().to_dict()


cutoffs, medians = get_cutoffs_and_medians()

# Pre-fill with the current assessment if there is one, otherwise dataset medians.
current = st.session_state.get("assessment", {}).get("values", {})
start = {f: current.get(f, medians[f]) for f in FEATURES}

LABELS = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)", "ph": "pH",
    "temperature": "Temperature", "humidity": "Humidity", "rainfall": "Rainfall",
}


def field(name: str):
    """One number input with range, unit and step taken from VALID_RANGES."""
    lo, hi, unit = VALID_RANGES[name]
    is_int = name in ("N", "P", "K")
    label = f"{LABELS[name]} ({unit})" if unit else LABELS[name]
    return st.number_input(
        label,
        min_value=float(lo), max_value=float(hi),
        value=float(round(start[name], 1)),
        step=1.0 if is_int else 0.1,
        format="%.0f" if is_int else "%.1f",
        key=f"in_{name}",
    )


with st.form("soil_form"):
    st.subheader("Soil nutrients and acidity")
    a, b, c, d = st.columns(4)
    with a: n = field("N")
    with b: p = field("P")
    with c: k = field("K")
    with d: ph = field("ph")

    st.subheader("Weather")
    e, f_, g = st.columns(3)
    with e: temperature = field("temperature")
    with f_: humidity = field("humidity")
    with g: rainfall = field("rainfall")

    submitted = st.form_submit_button("Save assessment", type="primary")

if submitted:
    values = {"N": n, "P": p, "K": k, "ph": ph,
              "temperature": temperature, "humidity": humidity, "rainfall": rainfall}

    errors = validate_inputs(values)  # range check (inputs are also range-limited by the widget)
    if errors:
        for msg in errors:
            st.error(msg)
    else:
        categories = {f: categorize(values[f], cutoffs[f]) for f in FEATURES}
        assessment_id = save_assessment(values)  # database write
        # Share with the other pages
        st.session_state["assessment"] = {
            "id": assessment_id, "values": values, "categories": categories,
        }
        # The What-if Lab will use this as its starting point / reset target.
        st.session_state.pop("whatif_values", None)
        st.success(f"Assessment #{assessment_id} saved.")

# ---- Show the current assessment (also after navigating back to this page)
assessment = st.session_state.get("assessment")
if assessment:
    st.subheader(f"Current assessment (#{assessment['id']})")
    cols = st.columns(len(FEATURES))
    for col, f in zip(cols, FEATURES):
        unit = VALID_RANGES[f][2]
        lo_cut, hi_cut = cutoffs[f]
        col.metric(LABELS[f].split(" (")[0], f"{assessment['values'][f]:g} {unit}".strip(),
                   help=f"low < {lo_cut:.1f} <= medium < {hi_cut:.1f} <= high")
        col.markdown(
            f"<div class='card' style='text-align:center;padding:0.4rem'>"
            f"{assessment['categories'][f]}</div>",
            unsafe_allow_html=True,
        )
    st.caption("Hover a value to see the low / medium / high cut-offs used. "
               "This assessment is now available to the Crop Advisor, What-if Lab and Cluster Explorer.")
else:
    st.caption("No assessment yet. Fill in the form and save to continue.")
