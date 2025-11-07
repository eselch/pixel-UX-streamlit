import streamlit as st
import common as ui
import data_processing as dp
from curve_editor import show_curve_plot, show_curve_controls, get_interpolated_curve_lut

ui.apply_base_ui("Configure Your Mattress")

# Initialize session state if not already done
if "answers" not in st.session_state:
    st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}

# Initialize master arrays from data_processing
dp.initialize_sleeper_master_arrays()

st.session_state.setdefault("current_configure_sleeper", "Sleeper 1")

#Column Layout
col1, col2 = st.columns([3, 1])

# Render controls first (col2) to ensure updates happen before plot
with col2:
    # Sleeper selection
    sleepers_available = []
    if st.session_state.answers.get("sleeper_1"):
        sleepers_available.append("Sleeper 1")
    if st.session_state.get("show_right") and st.session_state.answers.get("sleeper_2"):
        sleepers_available.append("Sleeper 2")

    if len(sleepers_available) > 1:
        current_selection = st.session_state.get("current_configure_sleeper", "Sleeper 1")
        
        # Two-column button layout for sleeper selection
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button(
                "Sleeper 1",
                use_container_width=True,
                type="primary" if current_selection == "Sleeper 1" else "secondary",
                key="sleeper_1_btn"
            ):
                st.session_state["current_configure_sleeper"] = "Sleeper 1"
                st.rerun()
        
        with btn_col2:
            if st.button(
                "Sleeper 2",
                use_container_width=True,
                type="primary" if current_selection == "Sleeper 2" else "secondary",
                key="sleeper_2_btn"
            ):
                st.session_state["current_configure_sleeper"] = "Sleeper 2"
                st.rerun()
    else:
        default_sleeper = sleepers_available[0] if sleepers_available else "Sleeper 1"
        st.session_state["current_configure_sleeper"] = default_sleeper

    current_sleeper = st.session_state["current_configure_sleeper"]
    side_key = "sleeper_1" if current_sleeper == "Sleeper 1" else "sleeper_2"

    # Bed size selection
    selected_bed = st.selectbox(
        "Bed size",
        options=list(dp.BED_SIZES.keys()),
        index=list(dp.BED_SIZES.keys()).index(st.session_state.get("bed_size", "Queen")),
        label_visibility="collapsed",
        key="bed_size_selector",
        on_change=lambda: dp.set_bed_size(st.session_state.bed_size_selector)
    )
    
    # Base firmness selection with callback
    current_firmness = st.session_state.answers.get(side_key, {}).get("firmness_value", 2)
    
    def on_firmness_change():
        """Callback when base firmness changes - shifts entire master array"""
        selected_firmness = st.session_state[f"firmness_selector_{side_key}"]
        old_firmness = st.session_state.answers[side_key].get("firmness_value", 2)
        
        if selected_firmness != old_firmness:
            # Calculate the shift amount
            shift = selected_firmness - old_firmness
            
            # Get current master array and shift all values
            master_array_current = dp.get_master_array(side_key)
            shifted_array = master_array_current + shift
            shifted_array = shifted_array.clip(0, 4)  # Keep within valid range
            
            # Update session state
            st.session_state.answers[side_key]["firmness_value"] = selected_firmness
            dp.set_master_array(side_key, shifted_array)
    
    selected_firmness = st.selectbox(
        "Base Firmness",
        options=dp.FIRMNESS_OPTIONS,
        format_func=lambda x: dp.FIRMNESS_LABELS[x],
        index=dp.FIRMNESS_OPTIONS.index(current_firmness) if current_firmness in dp.FIRMNESS_OPTIONS else 2,
        label_visibility="collapsed",
        key=f"firmness_selector_{side_key}",
        on_change=on_firmness_change
    )

    st.write("")  # adds spacing

    # Show curve controls
    show_curve_controls(side_key=side_key)

with col1:
    current_sleeper = st.session_state["current_configure_sleeper"]
    side_key = "sleeper_1" if current_sleeper == "Sleeper 1" else "sleeper_2"

    # Display sleeper name from session state or default to current sleeper
    sleeper_name = st.session_state.answers.get(side_key, {}).get("setting1", current_sleeper)
    st.subheader(sleeper_name)
    
    # Generate high-resolution LUT from interpolated curve for pixel map
    # This uses the smooth Hermite curve instead of the master array
    curve_lut = get_interpolated_curve_lut(side_key)
    width = dp.get_array_width()
    pixel_map_2d = dp.pixel_map(curve_lut, width)

    # Custom colorscale: maps firmness values 0-4 to specific Pantone colors
    # 0 = Very Soft (Teal), 1 = Soft (Tibetan Stone), 2 = Medium (White),
    # 3 = Firm (Yolk Yellow), 4 = Very Firm (Mandarin Red)
    custom_colorscale = [
        [0.0, "#E9F1F0"],
        [0.25, "#A7C7BF"],
        [0.5, "#1E99A8"],
        [0.75, "#006261"],
        [1.0, "#0A2734"]
    ]

    # Draw pixel map with max height and custom colors
    dp.draw_pixel_map(pixel_map_2d, height=450, colorscale=custom_colorscale)
    
    # Show curve plot with custom height - let it fetch fresh data
    show_curve_plot(side_key=side_key, height=200)# Debug: Display collected data

st.markdown("---")
st.subheader("Debug: Session Data")
st.json(st.session_state.answers)


def go_prev():
    st.switch_page("pages/2_Mapping.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label=None, right_action=None)
