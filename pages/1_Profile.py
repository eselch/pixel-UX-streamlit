import streamlit as st
import common as ui
from survey_questions import render_survey_questions

ui.apply_base_ui("Create Your Sleep Profile")

st.write("Welcome to the Pixel Mattress. Please enter the following information.")

st.write("")  # adds one line of space
st.write("")  # adds another line

# initialize session state if not already done
if "answers" not in st.session_state:
    st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}

if "show_right" not in st.session_state:
    st.session_state.show_right = False

# Create columns for layout
col_left, _, col_right = st.columns([3, 1, 3])

with col_left:
    # Sleeper 1 profile header and questions
    _, col_center_left, _ = st.columns([1, 2, 1])
    with col_center_left:
        st.header("SLEEPER 1")
    render_survey_questions("sleeper_1")

with col_right:
    if st.session_state.show_right:
        # Sleeper 2 profile header and questions
        _, col_center_right, _ = st.columns([1, 2, 1])
        with col_center_right:
            st.header("SLEEPER 2")
        render_survey_questions("sleeper_2")
    else:
        # "Add Sleeper" button
        st.write("")
        st.write("")
        st.write("")
        if st.button("Add Sleeper", use_container_width=True, key="add_partner_btn"):
            st.session_state.show_right = True
            st.rerun()

st.markdown("---")
st.write("Use the sidebar or the NEXT button below to move through the app.")



def go_mapping():
    st.switch_page("pages/2_Mapping.py")

ui.nav_row(left_label=None, left_action=None, right_label="NEXT", right_action=go_mapping)


