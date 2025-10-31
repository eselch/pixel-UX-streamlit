import streamlit as st
import common as ui

ui.apply_base_ui("Mapping")

# Initialize session state if not already done
if "answers" not in st.session_state:
    st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}

st.write("Upload pressure map CSV files for your profiles.")

st.write("")  # adds spacing
st.write("")

# Render dual column headers
col_left, col_right = ui.render_dual_column_headers("Sleeper 1", "Sleeper 2")

with col_left:
    st.subheader("Upload Pressure Map")
    left_file = st.file_uploader(
        "Choose CSV file",
        type="csv",
        key="left_csv",
        label_visibility="collapsed"
    )
    if left_file:
        st.session_state.answers["sleeper_1"]["csv_file"] = left_file
        st.success("File uploaded!")

with col_right:
    st.subheader("Upload Pressure Map")
    right_file = st.file_uploader(
        "Choose CSV file",
        type="csv",
        key="right_csv",
        label_visibility="collapsed"
    )
    if right_file:
        st.session_state.answers["sleeper_2"]["csv_file"] = right_file
        st.success("File uploaded!")

st.markdown("---")

# Debug: Display collected data
st.markdown("---")
st.subheader("Debug: Session Data")
st.json(st.session_state.answers)

def go_prev():
    st.switch_page("pages/1_Profile.py")

def go_next():
    st.switch_page("pages/3_Configure.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label="NEXT", right_action=go_next)
