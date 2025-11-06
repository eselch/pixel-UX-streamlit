"""Survey questions module for the Profile page."""

import streamlit as st
import data_processing as dp


def render_survey_questions(side_key: str):
    """Render survey questions for a given side (left or right).
    
    Args:
        side_key: "left" or "right" - used for session_state and widget keys
    """
# Question 1: Name
    st.subheader("First Name")
    st.session_state.answers[side_key]["setting1"] = st.text_input(
        "Enter your first name:", 
        value=st.session_state.answers[side_key].get("setting1", ""),
        label_visibility="collapsed",
        key=f"{side_key}_setting1"
    )
# Question 2: Height 
    st.subheader("Height (Inches)")
    height_input = st.text_input(
        "Enter your height:", 
        value=st.session_state.answers[side_key].get("setting2", ""),
        label_visibility="collapsed",
        key=f"{side_key}_setting2"
    )
    
    # Validate height input
    if height_input:
        try:
            height_value = float(height_input)
            if height_value < 40:
                st.error("Height must be at least 40 inches")
            else:
                st.session_state.answers[side_key]["setting2"] = height_input
        except ValueError:
            st.error("Height must contain only numbers")
    else:
        st.session_state.answers[side_key]["setting2"] = height_input
    
    st.subheader("Weight (Lbs)")
    st.session_state.answers[side_key]["setting3"] = st.text_input(
        "Enter your weight:", 
        value=st.session_state.answers[side_key].get("setting3", ""),
        label_visibility="collapsed",
        key=f"{side_key}_setting3"
    )
# Question 4: Sleep Position
    st.subheader("Sleep Position")
    col1, col2, col3 = st.columns(3)
    
    # Initialize selected positions if not already in session_state
    if f"{side_key}_positions" not in st.session_state:
        st.session_state[f"{side_key}_positions"] = []
    
    selected_positions = []
    
    with col1:
        if st.button("Front", key=f"{side_key}_front_btn"):
            if "Front" not in st.session_state[f"{side_key}_positions"]:
                st.session_state[f"{side_key}_positions"].append("Front")
            else:
                st.session_state[f"{side_key}_positions"].remove("Front")
    
    with col2:
        if st.button("Side", key=f"{side_key}_side_btn"):
            if "Side" not in st.session_state[f"{side_key}_positions"]:
                st.session_state[f"{side_key}_positions"].append("Side")
            else:
                st.session_state[f"{side_key}_positions"].remove("Side")
    
    with col3:
        if st.button("Back", key=f"{side_key}_back_btn"):
            if "Back" not in st.session_state[f"{side_key}_positions"]:
                st.session_state[f"{side_key}_positions"].append("Back")
            else:
                st.session_state[f"{side_key}_positions"].remove("Back")
    
    # Store selected positions in answers
    st.session_state.answers[side_key]["sleep_positions"] = st.session_state[f"{side_key}_positions"]
    
    # Display selected positions
    if st.session_state.answers[side_key]["sleep_positions"]:
        st.caption(f"Selected: {', '.join(st.session_state.answers[side_key]['sleep_positions'])}")

# Question 5: Mattress Firmness
    st.subheader("Preferred Mattress Firmness")
    firmness_options = {
        "Super Soft": 0,
        "Soft": 1,
        "Medium": 2,
        "Firm": 3,
        "Extra Firm": 4
    }
    
    selected_firmness_label = st.selectbox(
        "Select your preferred firmness:",
        options=list(firmness_options.keys()),
        index=list(firmness_options.keys()).index(st.session_state.answers[side_key].get("firmness_label", "Medium")),
        label_visibility="collapsed",
        key=f"{side_key}_firmness"
    )
    
    firmness_value = firmness_options[selected_firmness_label]
    st.session_state.answers[side_key]["firmness_value"] = firmness_value
    st.session_state.answers[side_key]["firmness_label"] = selected_firmness_label
    
    # Trigger master array update when firmness changes
    dp.update_master_array_from_firmness(side_key, firmness_value)

# Question 6: Gender
    st.subheader("Gender")
    st.session_state.answers[side_key]["gender"] = st.radio(
        "Select your gender:",
        options=["Male", "Female", "Other"],
        index=["Male", "Female", "Other"].index(st.session_state.answers[side_key].get("gender", "Male")),
        label_visibility="collapsed",
        key=f"{side_key}_gender",
        horizontal=True
    )