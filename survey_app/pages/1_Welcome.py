import streamlit as st
import common as ui
from curve_editor import show_curve_editor


ui.apply_base_ui("Welcome")

lut = show_curve_editor(key="tonecurve", sample_count=256, initial=3)
# lut is a list of 256 ints (0-255). Stored in st.session_state['tonecurve_lut'] too.

st.write("Welcome! Please answer the following questions honestly.")
st.write("Click NEXT to begin.")

def go_next():
    st.switch_page("pages/2_Questions.py")

ui.nav_row(left_label=None, left_action=None, right_label="NEXT", right_action=go_next)
