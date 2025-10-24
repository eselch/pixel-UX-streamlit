import streamlit as st
import common as ui

ui.apply_base_ui("Review Your Responses")

answers = st.session_state.get("answers", {})
if not answers:
    st.warning("No answers found. Please go back to fill the survey.")
else:
    st.write("### Your Answers")
    for q, a in answers.items():
        st.write(f"- **{q}**: {a}")

def go_prev():
    st.switch_page("pages/2_Questions.py")

def go_next():
    st.switch_page("pages/4_Thank_You.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label="NEXT", right_action=go_next)
