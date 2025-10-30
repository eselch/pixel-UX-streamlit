import streamlit as st
import common as ui

ui.apply_base_ui("Get Started")

st.write("Welcome to Pixel UX. This interactive survey will help us understand your preferences.")
st.markdown("---")

if st.button("Start Survey", key="start_button", use_container_width=True):
    st.switch_page("pages/1_Profile.py")
