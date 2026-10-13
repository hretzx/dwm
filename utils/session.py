"""
utils/session.py
----------------
Small helper shared by the Crop Advisor and What-if Lab pages.

It reads the current soil assessment that the Soil Analysis page stored in
st.session_state and returns it as a clean {feature: float} dict.

NOTE: if your Soil Analysis page stores the assessment under a different
session_state key, change ASSESSMENT_KEY below (one line, nothing else).
"""
import streamlit as st

ASSESSMENT_KEY = "assessment"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def get_current_assessment() -> dict | None:
    """Return the current assessment or None if the user has not made one yet."""
    raw = st.session_state.get(ASSESSMENT_KEY)
    if not raw:
        return None
    lower = {str(k).lower(): v for k, v in dict(raw).items()}  # tolerate N vs n
    try:
        return {f: float(lower[f.lower()]) for f in FEATURES}
    except (KeyError, TypeError, ValueError):
        return None


def require_assessment() -> dict:
    """Show a friendly message and stop the page if there is no assessment."""
    a = get_current_assessment()
    if a is None:
        st.info("No soil assessment yet. Enter your soil values first.")
        if st.button("Go to Soil Analysis", type="primary"):
            st.switch_page("pages/1_Soil_Analysis.py")
        st.stop()
    return a
