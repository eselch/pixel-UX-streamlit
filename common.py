import streamlit as st

def apply_base_ui(title: str = ""):
    """Apply base UI: set page config and show optional logo and title.

    The app's color accents (sliders, toggles, etc.) are controlled by
    Streamlit's theme in `.streamlit/config.toml`. Keep this function
    minimal so theming is handled centrally by the theme file.
    """
    page_title = title if title else "Survey App"
    st.set_page_config(page_title=page_title, layout="wide")

    # logo (top-right) — optional, ignore errors if missing
    try:
        st.image("primary-logo.svg", width=110, caption=None)
    except Exception:
        pass

    if title:
        st.title(title)

    # custom CSS for buttons to match theme colors
    st.markdown("""
    <style>
        /* Default button styling (secondary) */
        .stButton > button {
            background-color: #F0F2F6 !important;
            color: #6e6f72 !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 0px !important;
            font-weight: bold !important;
            cursor: pointer !important;
        }
        .stButton > button:hover {
            background-color: #036d7c !important;
            color: white !important;
        }
        .stButton > button:active {
            background-color: #025763 !important;
            color: white !important;
        }
        
        /* Primary button styling (selected state) */
        .stButton > button[kind="primary"] {
            background-color: #036d7c !important;
            color: white !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #025763 !important;
            color: white !important;
        }
        
        /* Disabled button styling */
        .stButton > button:disabled {
            background-color: #036d7c !important;
            color: white !important;
            opacity: 1 !important;
            cursor: default !important;
        }
        
        body, p, span, div {
            font-size: 16px !important;
            font-weight: 350 !important;
        }
        h1 {
            font-size: 40px !important;
            margin-bottom: 0.1rem !important;
            font-weight: 250 !important;
        }
        h2 {
            font-size: 20px !important;
            margin-bottom: 0.25rem !important;
            font-weight: 700 !important;
            text-align: center !important;
        }
        h3 {
            font-size: 20px !important;
            font-weight: 350 !important;
            margin-bottom: 0.2rem !important;

        }
    </style>
    """, unsafe_allow_html=True)

def is_dual_survey():
    """Check if user entered data in both sleeper profiles."""
    answers = st.session_state.get("answers", {})
    sleeper_1_data = answers.get("sleeper_1", {})
    sleeper_2_data = answers.get("sleeper_2", {})
    
    # If sleeper_2 column has any non-empty data, it's a dual survey
    return bool(sleeper_2_data) and any(sleeper_2_data.values())

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

def render_dual_column_headers(left_header: str = "Left", right_header: str = "Right"):
    """Render centered headers in dual columns (left and right).
    
    Args:
        left_header: Header text for left column
        right_header: Header text for right column
    
    Returns:
        tuple: (col_left, col_right) for use with context managers
    """
    # Add narrow columns on outer edges to center content
    _, col_left, _, col_right, _ = st.columns([1, 2, 1, 2, 1])
    
    with col_left:
        _, col_center_left, _ = st.columns([1, 2, 1])
        with col_center_left:
            st.header(left_header)
    
    with col_right:
        _, col_center_right, _ = st.columns([1, 2, 1])
        with col_center_right:
            st.header(right_header)
    
    return col_left, col_right
