import streamlit as st
import common as ui

ui.apply_base_ui("Welcome to the Survey")

st.write("This is a lightweight, multi-page Streamlit survey framework you can expand.")
st.markdown("---")
st.write("Use the sidebar or the NEXT button below to move between pages.")

def go_intro():
    st.switch_page("pages/1_Welcome.py")

ui.nav_row(left_label=None, left_action=None, right_label="NEXT", right_action=go_intro)
