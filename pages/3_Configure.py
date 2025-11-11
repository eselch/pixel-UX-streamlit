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
    
    # Add swap sides button
    if st.button("Swap Sides", use_container_width=True, key="swap_sides_btn"):
        st.session_state["sleeper_1_on_left"] = not st.session_state["sleeper_1_on_left"]
        st.rerun()
    
    st.write("")  # adds spacing
    
    # Add remap range control (only show if CSV data exists)
    if "csv_data" in st.session_state and side_key in st.session_state.csv_data:
        st.markdown("**Pressure Map Range**")
        
        # Get current remap range from session state
        current_remap_range = st.session_state.answers.get(side_key, {}).get("remap_range", "high")
        
        # Create selectbox for remap range
        remap_range = st.selectbox(
            "Pressure Map Range",
            options=["low", "high"],
            format_func=lambda x: "Low (1-3)" if x == "low" else "High (0-4)",
            index=0 if current_remap_range == "low" else 1,
            label_visibility="collapsed",
            key=f"remap_range_selector_{side_key}"
        )
        
        # If remap range changed, reapply the pressure map
        if remap_range != current_remap_range:
            # Get the downsampled pressure map from session state
            if side_key in st.session_state.csv_data:
                sensel_data = st.session_state.csv_data[side_key]["sensel_data"]
                downsampled = dp.downsample_pressure_map(sensel_data, target_shape=(17, 9))
                
                # Get the current number of control points
                num_control_points = st.session_state.answers.get(side_key, {}).get("num_control_points", 6)
                
                # Reapply with new remap range
                dp.apply_pressure_map_to_curve(
                    downsampled, 
                    side_key=side_key, 
                    num_control_points=num_control_points,
                    remap_range=remap_range
                )
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
        
        # Display names above heatmap
        _, name_col1, name_col2, _ = st.columns(4)
        with name_col1:
            st.markdown(f"<h3 style='text-align: center;'>{left_name}</h3>", unsafe_allow_html=True)
        with name_col2:
            st.markdown(f"<h3 style='text-align: center;'>{right_name}</h3>", unsafe_allow_html=True)
        
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
    else:
        # Single sleeper mode - show same data on both sides
        sleeper_name = st.session_state.answers.get(side_key, {}).get("setting1", current_sleeper)
        st.markdown(f"<h3 style='text-align: center;'>{sleeper_name}</h3>", unsafe_allow_html=True)
        
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

    # Center the heatmap using CSS
    st.markdown("""
        <style>
        div[data-testid="stPlotlyChart"] {
            display: flex;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Draw pixel map with fixed width and custom colors
    dp.draw_pixel_map(pixel_map_2d, height=450, colorscale=custom_colorscale)
    
    # Show curve plot with custom height - let it fetch fresh data
    show_curve_plot(side_key=side_key, height=200)# Debug: Display collected data

st.markdown("---")


def go_prev():
    st.switch_page("pages/2_Mapping.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label=None, right_action=None)
