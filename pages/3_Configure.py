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

# Initialize sleeper side preference (which sleeper is on left vs right)
st.session_state.setdefault("sleeper_1_on_left", True)

#Column Layout
col1, col2 = st.columns([2, 1])

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
        
        # Get sleeper names for button labels
        sleeper_1_name = st.session_state.answers.get("sleeper_1", {}).get("setting1", "Sleeper 1")
        sleeper_2_name = st.session_state.answers.get("sleeper_2", {}).get("setting1", "Sleeper 2")
        
        # Two-column button layout for sleeper selection
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            # Apply selected styling via container
            is_selected = current_selection == "Sleeper 1"
            button_style = "primary" if is_selected else "secondary"
            
            if st.button(
                sleeper_1_name,
                use_container_width=True,
                type=button_style,
                key="sleeper_1_btn",
                disabled=is_selected  # Disable if already selected
            ):
                st.session_state["current_configure_sleeper"] = "Sleeper 1"
                st.rerun()
        
        with btn_col2:
            # Apply selected styling via container
            is_selected = current_selection == "Sleeper 2"
            button_style = "primary" if is_selected else "secondary"
            
            if st.button(
                sleeper_2_name,
                use_container_width=True,
                type=button_style,
                key="sleeper_2_btn",
                disabled=is_selected  # Disable if already selected
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
            
            # Shift master array directly
            master_array_current = dp.get_master_array(side_key)
            shifted_array = master_array_current + shift
            shifted_array = shifted_array.clip(0, 4)  # Keep within valid range
            dp.set_master_array(side_key, shifted_array)
            
            # Update session state firmness value
            st.session_state.answers[side_key]["firmness_value"] = selected_firmness
    
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
    
    # Add swap sides button
    if st.button("Swap Sides", use_container_width=True, key="swap_sides_btn"):
        st.session_state["sleeper_1_on_left"] = not st.session_state["sleeper_1_on_left"]
        st.rerun()
    
    st.write("")  # adds spacing

    # Show curve controls
    show_curve_controls(side_key=side_key)

with col1:
    current_sleeper = st.session_state["current_configure_sleeper"]
    side_key = "sleeper_1" if current_sleeper == "Sleeper 1" else "sleeper_2"

    # Check if we have both sleepers configured
    has_both_sleepers = (
        st.session_state.answers.get("sleeper_1") and 
        st.session_state.get("show_right") and 
        st.session_state.answers.get("sleeper_2")
    )

    if has_both_sleepers:
        # Determine which sleeper is on which side
        if st.session_state["sleeper_1_on_left"]:
            left_key = "sleeper_1"
            right_key = "sleeper_2"
        else:
            left_key = "sleeper_2"
            right_key = "sleeper_1"
        
        # Get names for labels
        left_name = st.session_state.answers.get(left_key, {}).get("setting1", "Sleeper " + left_key[-1])
        right_name = st.session_state.answers.get(right_key, {}).get("setting1", "Sleeper " + right_key[-1])
        
        # Generate interpolated curves for both sleepers
        left_lut = get_interpolated_curve_lut(left_key)
        right_lut = get_interpolated_curve_lut(right_key)
        
        # Ensure LUTs are valid before proceeding
        if left_lut is None or right_lut is None:
            st.error("Error generating curve data. Please go back to Profile page.")
            st.stop()
        
        # Get width per sleeper (half the total bed width)
        total_width = dp.get_array_width()
        width_per_sleeper = total_width // 2
        
        # Create dual sleeper pixel map
        pixel_map_2d = dp.pixel_map_dual_sleeper(left_lut, right_lut, width_per_sleeper)
        heatmap_title = f"{left_name}  |  {right_name}"
    else:
        # Single sleeper mode - show same data on both sides
        sleeper_name = st.session_state.answers.get(side_key, {}).get("setting1", current_sleeper)
        
        # Generate high-resolution LUT from interpolated curve for pixel map
        curve_lut = get_interpolated_curve_lut(side_key)
        
        # Ensure LUT is valid before proceeding
        if curve_lut is None:
            st.error("Error generating curve data. Please go back to Profile page.")
            st.stop()
        
        # Get width per sleeper (half the total bed width)
        total_width = dp.get_array_width()
        width_per_sleeper = total_width // 2
        
        # Use the same data for both sides
        pixel_map_2d = dp.pixel_map_dual_sleeper(curve_lut, curve_lut, width_per_sleeper)
        heatmap_title = sleeper_name

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

    # Center the heatmap and curve plot using Streamlit column layout (works on Cloud)
    heatmap_cols = st.columns([1, 7, 1])
    with heatmap_cols[1]:
        dp.draw_pixel_map(
            pixel_map_2d,
            height=500,
            colorscale=custom_colorscale,
            title=heatmap_title,
        )

    curve_cols = st.columns([1, 7, 1])
    with curve_cols[1]:
        show_curve_plot(side_key=side_key, height=200, width=None)

st.markdown("---")

# Debug: Show session state
with st.expander("🔍 Debug: Session State"):
    import json
    st.json(st.session_state.to_dict())


def go_prev():
    st.switch_page("pages/2_Mapping.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label=None, right_action=None)
