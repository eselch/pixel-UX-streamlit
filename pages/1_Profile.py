import streamlit as st
import common as ui

ui.apply_base_ui("Create Your Sleep Profile")

st.write("Welcome to the Pixel Mattress. Please enter the following information.")

if "answers" not in st.session_state:
    st.session_state.answers = {"left": {}, "right": {}}

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Left Profile")
    ui.render_survey_questions("left")

with col_right:
    st.markdown("### Right Profile")
    ui.render_survey_questions("right")

st.markdown("---")
st.write("Use the sidebar or the NEXT button below to move through the app.")

def go_mapping():
    st.switch_page("pages/2_Mapping.py")

ui.nav_row(left_label=None, left_action=None, right_label="NEXT", right_action=go_mapping)


