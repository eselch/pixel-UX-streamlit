import streamlit as st
import common as ui
import data_processing as dp
from curve_editor import show_curve_editor

ui.apply_base_ui("Configure")

# Initialize session state if not already done
if "answers" not in st.session_state:
    st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}

# Initialize master arrays from data_processing
dp.initialize_sleeper_master_arrays()

# Add sleeper toggle at top
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    sleepers_available = []
    if st.session_state.answers.get("sleeper_1"):
        sleepers_available.append("Sleeper 1")
    if st.session_state.get("show_right") and st.session_state.answers.get("sleeper_2"):
        sleepers_available.append("Sleeper 2")
    
    if len(sleepers_available) > 1:
        current_sleeper = st.session_state.get("current_configure_sleeper", sleepers_available[0])
        st.radio(
            "Select Sleeper",
            options=sleepers_available,
            index=sleepers_available.index(current_sleeper) if current_sleeper in sleepers_available else 0,
            horizontal=True,
            label_visibility="collapsed",
            key="sleeper_toggle",
            on_change=lambda: st.session_state.update({"current_configure_sleeper": st.session_state["sleeper_toggle"]})
        )
    else:
        st.session_state["current_configure_sleeper"] = "Sleeper 1"

# Map selected sleeper to side_key
current_sleeper = st.session_state.get("current_configure_sleeper", "Sleeper 1")
side_key = "sleeper_1" if current_sleeper == "Sleeper 1" else "sleeper_2"

# Display curve editor for selected sleeper
show_curve_editor(side_key=side_key)

def go_prev():
    st.switch_page("pages/2_Mapping.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label=None, right_action=None)
