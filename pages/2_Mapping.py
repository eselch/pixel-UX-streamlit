import streamlit as st
import common as ui


ui.apply_base_ui("Mapping")

st.write("Click NEXT to continue.")

def go_prev():
    st.switch_page("pages/1_Profile.py")

def go_next():
    st.switch_page("pages/3_Configure.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label="NEXT", right_action=go_next)
