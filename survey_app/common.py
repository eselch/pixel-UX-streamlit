import streamlit as st

def apply_base_ui(title: str = ""):
    """Apply base UI: set page config and show optional logo and title.

    The app's color accents (sliders, toggles, etc.) are controlled by
    Streamlit's theme in `.streamlit/config.toml`. Keep this function
    minimal so theming is handled centrally by the theme file.
    """
    st.set_page_config(page_title="Survey App", layout="wide")

    # logo (top-left) — optional, ignore errors if missing
    try:
        st.image("primary-logo.svg", width=96, caption=None)
    except Exception:
        pass

    if title:
        st.title(title)

def nav_row(left_label=None, left_action=None, right_label=None, right_action=None):
    """Draw a two-button row at the bottom-ish area."""
    st.markdown('<div class="nav-area"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if left_label and st.button(left_label, key=f"prev_{left_label}"):
            if left_action:
                left_action()
    with col2:
        # place on the right by inserting spacer then the button
        c_sp, c_btn = st.columns([3, 1])
        with c_btn:
            if right_label and st.button(right_label, key=f"next_{right_label}"):
                if right_action:
                    right_action()
