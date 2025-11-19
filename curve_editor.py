"""Curve editor widget for Streamlit apps.

Provides show_curve_editor() which renders a cubic curve
through the master 1D array stored in session state via data_processing.

Usage (from any page):
    from curve_editor import show_curve_editor
    show_curve_editor()  # Uses array_length from session state
    show_curve_editor(side_key="sleeper_2")

The function reads/writes to:
    st.session_state.answers[side_key]["master_array"]
"""
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go

import data_processing as dp


def _is_spline_mode(side_key: str = "sleeper_1") -> bool:
    """Check if the curve editor is in scipy spline mode (vs manual Hermite mode).
    
    Spline mode is enabled when pressure map data has been uploaded and fitted
    with a scipy spline. In this mode, control points are the spline's knots.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    
    Returns
    -------
    bool
        True if using scipy spline mode, False for manual Hermite mode
    """
    if "answers" not in st.session_state:
        return False
    
    if side_key not in st.session_state.answers:
        return False
    
    return st.session_state.answers[side_key].get("use_scipy_spline", False)


def _piecewise_hermite_smooth(xs: np.ndarray, ys: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Piecewise cubic Hermite interpolation with smooth tangents at control points.
    
    For interior points, the tangent is computed as the slope between neighboring points
    (finite difference method) to ensure smooth curves without flat spots.
    For endpoints, the tangent is set to the slope to the adjacent point.

    xs: 1D increasing x positions (length n)
    ys: 1D y values (length n)
    x_eval: points to evaluate (1D)

    Returns y values at x_eval.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    
    n = len(xs)
    
    # Compute tangents (derivatives) at each control point
    tangents = np.zeros(n)
    
    # First point: use slope to next point
    tangents[0] = (ys[1] - ys[0]) / (xs[1] - xs[0])
    
    # Interior points: use average of slopes to neighbors (finite difference)
    for i in range(1, n - 1):
        # Slope from previous to next point
        tangents[i] = (ys[i + 1] - ys[i - 1]) / (xs[i + 1] - xs[i - 1])
    
    # Last point: use slope from previous point
    tangents[-1] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    
    # Prepare output
    y_out = np.empty_like(x_eval, dtype=float)

    # Handle values outside the range
    left_mask = x_eval <= xs[0]
    right_mask = x_eval >= xs[-1]
    mid_mask = ~(left_mask | right_mask)

    y_out[left_mask] = ys[0]
    y_out[right_mask] = ys[-1]

    # Interpolate interior points using cubic Hermite spline
    if np.any(mid_mask):
        xm = x_eval[mid_mask]
        # Find segment for each evaluation point
        idx = np.searchsorted(xs, xm, side='right') - 1
        idx = np.clip(idx, 0, n - 2)

        x0 = xs[idx]
        x1 = xs[idx + 1]
        y0 = ys[idx]
        y1 = ys[idx + 1]
        m0 = tangents[idx]
        m1 = tangents[idx + 1]
        
        # Normalized parameter t ∈ [0, 1]
        h = x1 - x0
        t = (xm - x0) / h
        
        # Cubic Hermite basis functions
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        
        # Hermite interpolation with computed tangents
        yvals = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
        y_out[mid_mask] = yvals

    return y_out


def get_interpolated_curve_lut(
    side_key: str = "sleeper_1",
    array_length: int = None,
    num_points: int = 6,
    value_range: Tuple[int, int] = (0, 4),
) -> np.ndarray:
    """Generate a downsampled LUT array from the high-resolution interpolated curve.
    
    This creates a 1D array by evaluating the curve at each integer position from
    1 to array_length. The interpolation method depends on the mode:
    - Manual mode: Uses piecewise Hermite spline with fixed control points
    - Spline mode: Uses scipy UnivariateSpline rebuilt from edited knots
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    array_length : int, optional
        Length of the output LUT array. If None, retrieves from session state.
    num_points : int
        Number of control points used for interpolation (ignored in spline mode)
    value_range : Tuple[int, int]
        Allowed y value range (min, max)
    
    Returns
    -------
    np.ndarray
        1D array of length array_length with interpolated firmness values
    """
    try:
        if array_length is None:
            array_length = dp.get_array_length()
        
        vmin, vmax = value_range
        x_domain = (1, array_length)
        
        # Get control point positions and values (mode-aware)
        xs, ys_float, xmin, xmax = _get_curve_data(side_key, array_length, x_domain, num_points)
        
        # Clip control point y values to value range
        ys_float = np.clip(ys_float, vmin, vmax)
        
        # Evaluate the curve at each integer position using Hermite interpolation
        # (consistent display regardless of mode)
        lut_x = np.arange(1, array_length + 1)
        lut_y = _piecewise_hermite_smooth(xs, ys_float, lut_x)
        
        # Clip and round to integers
        lut_y = np.clip(lut_y, vmin, vmax)
        lut_y = np.round(lut_y).astype(int)
        
        return lut_y
        
    except Exception as e:
        # Return None if there's any error - will be caught by caller
        print(f"Error in get_interpolated_curve_lut for {side_key}: {e}")
        return None


def _get_curve_data(
    side_key: str = "sleeper_1",
    array_length: int = None,
    x_domain: Tuple[int, int] = None,
    num_points: int = 6,
    master_array: np.ndarray = None,
) -> tuple:
    """Get control points and interpolated curve data without rendering.
    
    In manual mode: Returns evenly-spaced control points from the master array.
    In spline mode: Returns the scipy spline knot positions and values.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    array_length : int, optional
        Length of the master array. If None, retrieves from session state.
    x_domain : Tuple[int, int], optional
        Domain of x (min, max). If None, derived from array_length.
    num_points : int
        Number of control points (used in manual mode only)
    master_array : np.ndarray, optional
        The master array to use. If None, fetches from session state.

    Returns
    -------
    tuple
        (xs, ys, xmin, xmax) - control point positions (1-indexed) and values
    """
    if array_length is None:
        array_length = dp.get_array_length()
    
    if x_domain is None:
        x_domain = (1, array_length)
    
    xmin, xmax = x_domain
    
    # Initialize master arrays if not already done
    dp.initialize_sleeper_master_arrays(array_length=array_length)
    
    # Get the master array for this sleeper
    if master_array is None:
        master_array = dp.get_master_array(side_key, array_length=array_length)
    else:
        master_array = np.asarray(master_array)
    
    # === CHECK MODE ===
    if _is_spline_mode(side_key):
        # === SPLINE MODE: Return knot positions and values ===
        spline_knots = st.session_state.answers[side_key].get("spline_knots", {})
        knot_x = spline_knots.get("x", [])
        knot_y = spline_knots.get("y", [])
        
        if len(knot_x) > 0 and len(knot_x) == len(knot_y):
            xs = np.array(knot_x, dtype=int)
            ys = np.array(knot_y, dtype=float)
            return xs, ys, xmin, xmax
        else:
            # No valid knots, fall through to manual mode
            pass
    
    # === MANUAL MODE: Return body-part-based control points ===
    
    # Use body part control points based on sleeper height
    xs = dp.get_body_part_control_points(side_key, array_length=array_length)
    
    # Get y values from master array at control point positions (1-indexed)
    ys = np.array([master_array[x - 1] for x in xs], dtype=float)
    
    return xs, ys, xmin, xmax


def show_curve_controls(
    side_key: str = "sleeper_1",
    array_length: int = None,
    x_domain: Tuple[int, int] = None,
    value_range: Tuple[int, int] = (0, 4),
    num_points: int = 6,
    master_array: np.ndarray = None,
) -> None:
    """Render control point tuning widgets (increment/decrement per point).
    
    In manual mode: Updates master_array at control point positions.
    In spline mode: Updates spline_knots and rebuilds master_array from spline.
    
    Call this in your layout (e.g., in a column).
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    array_length : int, optional
        Length of the master array. If None, retrieves from session state.
    x_domain : Tuple[int, int], optional
        Domain of x (min, max). If None, derived from array_length.
    value_range : Tuple[int, int]
        Allowed y value range (min, max)
    num_points : int
        Number of control points to display (ignored in spline mode)
    master_array : np.ndarray, optional
        The master array to use. If None, fetches from session state.
    """
    if array_length is None:
        array_length = dp.get_array_length()

    # Always fetch fresh master array from session state for sliders
    master_array = dp.get_master_array(side_key, array_length=array_length)

    xs, ys_float, _, _ = _get_curve_data(side_key, array_length, x_domain, num_points, master_array)
    vmin, vmax = value_range
    
    # Actual number of control points (varies in spline mode)
    actual_num_points = len(xs)
    
    # Round y values to integers for display
    ys = np.round(ys_float).astype(int)
    ys = np.clip(ys, vmin, vmax)
    
    st.subheader("Firmness Tuning")

    # Get base firmness to calculate offsets (only used for display in manual mode)
    base_firmness = st.session_state.answers.get(side_key, {}).get("firmness_value", 2)
    
    updated = False
    is_spline = _is_spline_mode(side_key)
    
    # Track updated knot values for spline mode
    updated_knot_y = ys_float.copy()

    for i in range(actual_num_points):
        x_pos = int(xs[i])
        current_value = int(ys[i])
        offset = current_value - base_firmness  # Calculate offset from base
        
        # Get label based on mode
        if is_spline:
            # Spline mode: show point number and row
            label = f"Point {i+1} · Row {x_pos}"
        else:
            # Manual mode: show body part label
            body_part = dp.BODY_PART_LABELS.get(i+1, f"Point {i+1}")
            label = f"{body_part} · Row {x_pos}"

        col_label, col_minus, col_plus = st.columns([2.5, 1, 1], gap="small")
        with col_label:
            st.markdown(f"**{label}**")

        with col_minus:
            if st.button("−", key=f"{side_key}_decrease_{i}", use_container_width=True, type="secondary"):
                if current_value > vmin:
                    updated_knot_y[i] = current_value - 1
                    updated = True

        with col_plus:
            if st.button("\+", key=f"{side_key}_increase_{i}", use_container_width=True, type="secondary"):
                if current_value < vmax:
                    updated_knot_y[i] = current_value + 1
                    updated = True

    if updated:
        # Store updated knot Y values
        updated_knot_y = np.clip(updated_knot_y, vmin, vmax)
        
        if is_spline:
            st.session_state.answers[side_key]["spline_knots"]["y"] = updated_knot_y.tolist()
        
        # Rebuild master array using Hermite interpolation (same for both modes)
        x_all = np.arange(1, array_length + 1)
        master_array_new = _piecewise_hermite_smooth(xs, updated_knot_y, x_all)
        master_array_new = np.clip(master_array_new, vmin, vmax)
        master_array_new = np.round(master_array_new).astype(int)
        
        # Update master array
        dp.set_master_array(side_key, master_array_new)

    return None


def show_curve_plot(
    side_key: str = "sleeper_1",
    array_length: int = None,
    x_domain: Tuple[int, int] = None,
    value_range: Tuple[int, int] = (0, 4),
    width: int = 700,
    height: int = 320,
    num_points: int = 6,
    master_array: np.ndarray = None,
    return_fig: bool = False,
):
    """Render the curve plot visualization.
    
    In manual mode: Uses Hermite interpolation with blue circles for control points.
    In spline mode: Uses scipy spline interpolation with red diamonds for knots.
    
    Call this in your layout (e.g., in a column).
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    array_length : int, optional
        Length of the master array. If None, retrieves from session state.
    x_domain : Tuple[int, int], optional
        Domain of x (min, max). If None, derived from array_length.
    value_range : Tuple[int, int]
        Allowed y value range (min, max)
    width/height : int
        Size of the plotly figure
    num_points : int
        Number of control points to display (ignored in spline mode)
    master_array : np.ndarray, optional
        The master array to use. If None, fetches from session state.
    return_fig : bool
        If True, return the figure object instead of displaying it
    
    Returns
    -------
    go.Figure or None
        Plotly figure object if return_fig=True, otherwise None
    """
    if array_length is None:
        array_length = dp.get_array_length()
    
    xs, ys_float, xmin, xmax = _get_curve_data(side_key, array_length, x_domain, num_points, master_array)
    vmin, vmax = value_range
    
    # Determine mode
    is_spline = _is_spline_mode(side_key)
    
    # Clip y values
    ys_float = np.clip(ys_float, vmin, vmax)
    
    # === GENERATE SMOOTH CURVE FOR DISPLAY ===
    # Always use Hermite interpolation for consistent display regardless of mode
    plot_samples = 1024
    plot_x = np.linspace(xmin, xmax, plot_samples)
    plot_y = _piecewise_hermite_smooth(xs, ys_float, plot_x)
    
    plot_y = np.clip(plot_y, vmin, vmax)

    fig = go.Figure()
    
    # === ADD PRESSURE MAP HEATMAP UNDERLAY IF CSV IS LOADED ===
    if "csv_data" in st.session_state and side_key in st.session_state.csv_data:
        try:
            # Get the downsampled pressure map - use same length as current array
            sensel_data = st.session_state.csv_data[side_key]["sensel_data"]
            downsampled = dp.downsample_pressure_map(sensel_data, target_shape=(array_length, 9))
            
            # Convert to 1D using max per row
            pressure_1d = dp.pressure_map_to_1d_array(downsampled)
            
            # Map pressure to 0-4 range using fixed reference scale
            max_pressure_psi = 2.0  # Reference: 2 psi fills the full graph
            
            # Scale pressure data: 0 psi = 0, 2 psi = 4
            scaled = (pressure_1d / max_pressure_psi) * 4.0
            scaled = np.clip(scaled, 0, 4)  # Clip to graph range
            
            # Create x positions (1-indexed row numbers)
            pressure_x = np.arange(1, len(pressure_1d) + 1)
            
            # Map each scaled value to a color using the same gradient as the heatmap
            # Blue gradient: light blue (0) to dark blue (4)
            def value_to_color(val):
                """Map a value in range 0-4 to the blue gradient color."""
                # Normalize to 0-1 range
                normalized = val / 4.0
                
                # Define the color stops (same as heatmap)
                if normalized <= 0.25:
                    # Interpolate between 0.0 (#e3f2fd) and 0.25 (#90caf9)
                    t = normalized / 0.25
                    return f'rgba({int(227 + t * (144 - 227))}, {int(242 + t * (202 - 242))}, {int(253 + t * (249 - 253))}, 0.3)'
                elif normalized <= 0.5:
                    # Interpolate between 0.25 (#90caf9) and 0.5 (#42a5f5)
                    t = (normalized - 0.25) / 0.25
                    return f'rgba({int(144 + t * (66 - 144))}, {int(202 + t * (165 - 202))}, {int(249 + t * (245 - 249))}, 0.3)'
                elif normalized <= 0.75:
                    # Interpolate between 0.5 (#42a5f5) and 0.75 (#1e88e5)
                    t = (normalized - 0.5) / 0.25
                    return f'rgba({int(66 + t * (30 - 66))}, {int(165 + t * (136 - 165))}, {int(245 + t * (229 - 245))}, 0.3)'
                else:
                    # Interpolate between 0.75 (#1e88e5) and 1.0 (#0d47a1)
                    t = (normalized - 0.75) / 0.25
                    return f'rgba({int(30 + t * (13 - 30))}, {int(136 + t * (71 - 136))}, {int(229 + t * (161 - 229))}, 0.3)'
            
            # Create color array for each bar
            bar_colors = [value_to_color(v) for v in scaled]
            
            # Add pressure data as bar graph background (inverted, from top down)
            # Use negative values with base at vmax to make bars extend downward
            fig.add_trace(go.Bar(
                x=pressure_x,
                y=-scaled,  # Negative values
                base=vmax,  # Start bars from top of graph (y=4)
                name='Pressure Data',
                marker=dict(color=bar_colors),
                width=0.8,
                showlegend=False,
                hoverinfo='skip'
            ))
        except Exception as e:
            # Silently ignore errors in pressure data overlay
            pass
    
    # Set curve color and control point style based on mode
    if is_spline:
        curve_color = '#0492a8'
        marker_symbol = 'circle'
        marker_size = 8
        control_label = 'PT'
    else:
        curve_color = '#0492a8'  
        marker_symbol = 'circle'
        marker_size = 8
        control_label = 'PT'
    
    # Add smooth curve
    fig.add_trace(go.Scatter(
        x=plot_x,
        y=plot_y,
        mode='lines',
        name='Curve',
        line=dict(color=curve_color, width=2)
    ))
    
    # Add control points / knots
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys_float,
        mode='markers',
        name=f'{control_label}s',
        marker=dict(color=curve_color, size=marker_size, symbol=marker_symbol)
    ))
    
    # Add labels above each control point / knot
    for i, (x, y) in enumerate(zip(xs, ys_float)):
        # Get label based on mode
        if is_spline:
            # Spline mode: show point number
            label_text = f"{control_label} {i+1}"
        else:
            # Manual mode: show body part label
            label_text = dp.BODY_PART_LABELS.get(i+1, f"Point {i+1}")
        
        fig.add_annotation(
            x=x,
            y=y,
            text=label_text,
            showarrow=False,
            yshift=15,
            font=dict(size=12, color='#6e6f72'),
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(4, 146, 168, 0)',
            borderwidth=1,
            borderpad=2
        )
    
    # In spline mode, add body part location overlays
    if is_spline:
        body_part_positions = dp.get_body_part_control_points(side_key, array_length=array_length)
        for i, x_pos in enumerate(body_part_positions):
            body_part_label = dp.BODY_PART_LABELS.get(i+1, "")
            
            # Add vertical line for body part location
            fig.add_vline(
                x=x_pos,
                line_dash="solid",
                line_color="rgba(66, 165, 245, 0.4)",  # Semi-transparent blue
                line_width=2
            )
            
            # Add label at bottom of chart for body part
            fig.add_annotation(
                x=x_pos,
                y=vmin,
                text=body_part_label,
                showarrow=False,
                yshift=10,
                font=dict(size=10, color='rgba(66, 165, 245, 0.8)'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(66, 165, 245, 0.3)',
                borderwidth=1,
                borderpad=2
            )
    
    # Add custom gridlines offset by 0.5 to be between numbers
    for i in np.arange(int(xmin) + 0.5, int(xmax), 1):
        fig.add_vline(x=i, line_dash="solid", line_color="LightGray", line_width=1)
    
    # Update layout
    fig.update_layout(
        xaxis_title='Rows',
        yaxis_title='Firmness Level',
        hovermode='closest',
        height=height,
        width=width,
        xaxis=dict(range=[xmin - 0.5, xmax + 0.5], dtick=1, fixedrange=True, showgrid=False),
        yaxis=dict(range=[vmin, vmax], dtick=1, fixedrange=True, showgrid=True, gridcolor='LightGray'),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=20, b=40)  # Reduce top margin
    )
    
    if return_fig:
        return fig
    else:
        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})


def show_curve_editor(
    side_key: str = "sleeper_1",
    array_length: int = None,
    x_domain: Tuple[int, int] = None,
    value_range: Tuple[int, int] = (0, 4),
    width: int = 700,
    height: int = 320,
    num_points: int = 6,
) -> None:
    """Render a complete curve editor with plot and controls in a 2-column layout.
    
    This is a convenience wrapper. For custom layouts, use show_curve_plot() and
    show_curve_controls() directly.

    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    array_length : int, optional
        Length of the master array. If None, retrieves from session state.
    x_domain : Tuple[int, int], optional
        Domain of x (min, max). If None, derived from array_length.
    value_range : Tuple[int, int]
        Allowed y value range (min, max)
    width/height : int
        Size of the plotly figure
    num_points : int
        Number of control points to display
    """
    col_plot, col_controls = st.columns([3, 1])

    with col_controls:
        master_array = show_curve_controls(
            side_key,
            array_length,
            x_domain,
            value_range,
            num_points,
        )

    with col_plot:
        show_curve_plot(
            side_key,
            array_length,
            x_domain,
            value_range,
            width,
            height,
            num_points,
            master_array,
        )


if __name__ == "__main__":
    # simple local test when running this module directly (not necessary in app)
    if not st._is_running_with_streamlit:
        print("Run inside streamlit: streamlit run <app>")
