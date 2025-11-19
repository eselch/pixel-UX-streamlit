import streamlit as st
from datetime import datetime
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
        """Callback when base firmness changes - shifts entire master array and knots if in spline mode"""
        selected_firmness = st.session_state[f"firmness_selector_{side_key}"]
        old_firmness = st.session_state.answers[side_key].get("firmness_value", 2)
        
        if selected_firmness != old_firmness:
            # Calculate the shift amount
            shift = selected_firmness - old_firmness
            
            # Check if we're in spline mode
            is_spline_mode = st.session_state.answers[side_key].get("use_scipy_spline", False)
            
            if is_spline_mode:
                # === SPLINE MODE: Refit from original pressure data with new firmness offset ===
                # Get current scale percentage (default to 50%)
                current_scale_percent = st.session_state.answers[side_key].get("curve_scale_percent", 50.0)
                
                # Get the original firmness when data was uploaded (stored as base reference)
                # We'll calculate total offset from the original base
                original_firmness = st.session_state.answers[side_key].get("original_firmness", 2)
                total_offset = selected_firmness - original_firmness
                
                # Refit spline from original data with scale percentage and offset
                dp.refit_spline_from_original(
                    side_key=side_key,
                    scale_percent=current_scale_percent,
                    firmness_offset=total_offset
                )
            else:
                # === MANUAL MODE: Shift master array directly ===
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
    
    # Add curve scale control (only show if in spline mode / CSV data exists)
    if "csv_data" in st.session_state and side_key in st.session_state.csv_data:
        is_spline_mode = st.session_state.answers.get(side_key, {}).get("use_scipy_spline", False)
        
        if is_spline_mode:
            st.subheader("Firmness Range")
            
            # Get current scale percentage from session state
            current_scale_percent = st.session_state.answers.get(side_key, {}).get("curve_scale_percent", 50.0)
            
            # Create slider for curve scale (1% to 100%)
            curve_scale_percent = st.slider(
                "Firmness Range",
                min_value=1.0,
                max_value=100.0,
                value=current_scale_percent,
                step=1.0,
                format="%d%%",
                label_visibility="collapsed",
                key=f"curve_scale_slider_{side_key}",
                help="1% = nearly flat, 100% = fills full graph range (0-4)"
            )
            
            # If scale changed, rescale the curve
            if curve_scale_percent != current_scale_percent:
                # Get current firmness and calculate offset from original
                current_firmness = st.session_state.answers[side_key].get("firmness_value", 2)
                original_firmness = st.session_state.answers[side_key].get("original_firmness", 2)
                firmness_offset = current_firmness - original_firmness
                
                # Update stored scale percentage
                st.session_state.answers[side_key]["curve_scale_percent"] = curve_scale_percent
                
                # Refit spline from original data with new scale percentage and current offset
                dp.refit_spline_from_original(
                    side_key=side_key,
                    scale_percent=curve_scale_percent,
                    firmness_offset=firmness_offset
                )
                
                st.rerun()
        
        st.write("")  # adds spacing
        
        # Add smoothing factor slider
        if is_spline_mode:
            st.subheader("Curve Smoothing")
            
            # Get current smoothing factor from session state
            array_length = dp.get_array_length()
            default_smoothing = array_length * 0.05  # 5% default (conservative midpoint)
            current_smoothing = st.session_state.answers.get(side_key, {}).get("spline_smoothing", default_smoothing)
            
            # Create slider for smoothing factor (0 = exact fit, higher = smoother)
            # Display as percentage of array length for intuitive control
            smoothing_percent = (current_smoothing / array_length) * 100.0 if array_length > 0 else 5.0
            
            smoothing_percent = st.slider(
                "Curve Smoothing",
                min_value=0.01,
                max_value=25.00,
                value=smoothing_percent,
                step=0.5,
                format="%.1f%%",
                label_visibility="collapsed",
                key=f"smoothing_slider_{side_key}",
                help="0% = exact fit through data points, higher = smoother curve with fewer knots"
            )
            
            # Convert percentage back to smoothing factor
            new_smoothing = (smoothing_percent / 100.0) * array_length
            
            # If smoothing changed, refit the curve
            if abs(new_smoothing - current_smoothing) > 0.01:
                # Get current scale and firmness offset
                current_scale_percent = st.session_state.answers[side_key].get("curve_scale_percent", 50.0)
                current_firmness = st.session_state.answers[side_key].get("firmness_value", 2)
                original_firmness = st.session_state.answers[side_key].get("original_firmness", 2)
                firmness_offset = current_firmness - original_firmness
                
                # Refit spline with new smoothing factor
                dp.refit_spline_from_original(
                    side_key=side_key,
                    scale_percent=current_scale_percent,
                    firmness_offset=firmness_offset,
                    smoothing_factor=new_smoothing
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
    heatmap_cols = st.columns([10, 1])
    with heatmap_cols[0]:
        dp.draw_pixel_map(
            pixel_map_2d,
            height=500,
            colorscale=custom_colorscale,
            title=heatmap_title,
        )

    curve_cols = st.columns([10, 1])
    with curve_cols[0]:
        show_curve_plot(side_key=side_key, height=200, width=None)

st.markdown("---")

# Debug: Show session state
with st.expander("🔍 Debug: Session State"):
    import json
    st.json(st.session_state.to_dict())


def go_prev():
    st.switch_page("pages/2_Mapping.py")

def export_data():
    # Placeholder for export functionality
    st.write("Export functionality to be implemented")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label="EXPORT", right_action=export_data)
