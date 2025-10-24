import streamlit as st
import common as ui

ui.apply_base_ui("Thank You")

st.success("Your responses have been submitted.")
st.write("We appreciate your time!")

def go_prev():
    st.switch_page("pages/3_Review.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label=None, right_action=None)
