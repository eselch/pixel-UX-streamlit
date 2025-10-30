import streamlit as st
import common as ui
from curve_editor import show_curve_editor

ui.apply_base_ui("Configure")

lut = show_curve_editor(key="tonecurve", sample_count=256, initial=3)
# lut is a list of 256 ints (0-255). Stored in st.session_state['tonecurve_lut'] too.

def go_prev():
    st.switch_page("pages/2_Mapping.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label=None, right_action=None)
