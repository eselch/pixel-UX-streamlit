import streamlit as st

def apply_base_ui(title: str = ""):
    """Apply base UI: set page config and show optional logo and title.

    The app's color accents (sliders, toggles, etc.) are controlled by
    Streamlit's theme in `.streamlit/config.toml`. Keep this function
    minimal so theming is handled centrally by the theme file.
    """
    page_title = title if title else "Survey App"
    st.set_page_config(page_title=page_title, layout="wide")

    # logo (top-left) — optional, ignore errors if missing
    try:
        st.image("primary-logo.svg", width=110, caption=None)
    except Exception:
        pass

    if title:
        st.title(title)

    # custom CSS for buttons to match theme colors
    st.markdown("""
    <style>
        .stButton > button {
            background-color: #0492a8 !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 0px !important;
            font-weight: bold !important;
            cursor: pointer !important;
        }
        .stButton > button:hover {
            background-color: #036d7c !important;
        }
        .stButton > button:active {
            background-color: #025763 !important;
        }
        body, p, span, div {
            font-size: 16px !important;
        }
        h1 {
            font-size: 40px !important;
            margin-bottom: 0.5rem !important;
        }
        h2 {
            font-size: 28px !important;
            margin-bottom: 0.5rem !important;
        }
        h3 {
            font-size: 20px !important;
        }
    </style>
    """, unsafe_allow_html=True)

def is_dual_survey():
    """Check if user entered data in both left and right profiles."""
    answers = st.session_state.get("answers", {})
    left_data = answers.get("left", {})
    right_data = answers.get("right", {})
    
    # If right column has any non-empty data, it's a dual survey
    return bool(right_data) and any(right_data.values())

def render_survey_questions(side_key: str):
    """Render survey questions for a given side (left or right).
    
    Args:
        side_key: "left" or "right" - used for session_state and widget keys
    """
    st.subheader("1) Configuration Setting 1")
    st.session_state.answers[side_key]["setting1"] = st.radio(
        "Select one:", 
        ["Option A", "Option B", "Option C", "Option D", "Option E"],
        index=2,
        label_visibility="collapsed",
        key=f"{side_key}_setting1"
    )
    
    st.subheader("2) Configuration Setting 2")
    st.session_state.answers[side_key]["setting2"] = st.slider(
        "Scale from 0 to 10", 0, 10, 5, key=f"{side_key}_setting2_slider", label_visibility="collapsed"
    )

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
