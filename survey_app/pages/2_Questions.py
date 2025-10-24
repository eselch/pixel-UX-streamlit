import streamlit as st
import common as ui

ui.apply_base_ui("Survey Questions")

if "answers" not in st.session_state:
    st.session_state.answers = {}

st.subheader("1) How satisfied are you with our product?")
st.session_state.answers["q1"] = st.radio(
    "Select one:", 
    ["Very Satisfied", "Satisfied", "Neutral", "Unsatisfied", "Very Unsatisfied"],
    index=2,
    label_visibility="collapsed"
)

st.subheader("2) How likely are you to recommend us?")
st.session_state.answers["q2"] = st.slider(
    "0 = not at all, 10 = definitely", 0, 10, 5, key="q2_slider", label_visibility="collapsed"
)

def go_prev():
    st.switch_page("pages/1_Welcome.py")

def go_next():
    st.switch_page("pages/3_Review.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label="NEXT", right_action=go_next)
