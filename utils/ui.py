"""
utils/ui.py
-----------
Theme helpers: a little custom CSS and the sidebar branding.
The base colours come from .streamlit/config.toml; this file only adds
rounded cards and spacing.
"""
import streamlit as st

PRIMARY = "#2E7D32"
LIGHT = "#E8F5E9"
ACCENT = "#66BB6A"
TEXT = "#333333"

_CSS = f"""
<style>
  .block-container {{ padding-top: 2.5rem; max-width: 1100px; }}
  h1, h2, h3 {{ color: {PRIMARY}; font-weight: 600; }}

  /* Sidebar: light green, brand shown above the page navigation */
  [data-testid="stSidebar"] {{ background-color: {LIGHT}; }}
  [data-testid="stSidebarContent"] {{ display: flex; flex-direction: column; }}
  [data-testid="stSidebarNav"] {{ order: 2; }}
  [data-testid="stSidebarUserContent"] {{ order: 1; padding-bottom: 0; }}
  .brand-name {{ color: {PRIMARY}; font-size: 1.7rem; font-weight: 700; margin: 0; }}
  .brand-tag  {{ color: {TEXT}; font-size: 0.9rem; margin: 0 0 0.5rem 0; opacity: 0.8; }}

  /* Rounded metric cards */
  [data-testid="stMetric"] {{
      background: {LIGHT}; border: 1px solid #C8E6C9; border-radius: 14px;
      padding: 1rem 1.25rem;
  }}
  [data-testid="stMetricValue"] {{ color: {PRIMARY}; }}

  /* Buttons */
  .stButton > button {{ border-radius: 10px; border: 1px solid {PRIMARY}; }}
  .stButton > button[kind="primary"] {{ background: {PRIMARY}; color: white; }}
  .stButton > button[kind="primary"]:hover {{ background: {ACCENT}; border-color: {ACCENT}; }}

  /* Soft card helper for custom HTML blocks */
  .card {{ background: {LIGHT}; border-radius: 14px; padding: 1.25rem 1.5rem; }}
</style>
"""


def apply_theme() -> None:
    """Inject CSS and draw the sidebar brand. Call once from app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p class="brand-name">AgriMine</p><p class="brand-tag">Precision Agriculture</p>',
        unsafe_allow_html=True,
    )


def page_header(title: str, caption: str) -> None:
    """Consistent page title + plain-language caption explaining the algorithm."""
    st.title(title)
    st.caption(caption)
