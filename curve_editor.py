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
    
    This creates a 1D array by evaluating the smooth Hermite curve at each integer
    position from 1 to array_length. This is the high-resolution version of the
    curve shown in the curve editor, suitable for driving the pixel map.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    array_length : int, optional
        Length of the output LUT array. If None, retrieves from session state.
    num_points : int
        Number of control points used for interpolation
    value_range : Tuple[int, int]
        Allowed y value range (min, max)
    
    Returns
    -------
    np.ndarray
        1D array of length array_length with interpolated firmness values
    """
    if array_length is None:
        array_length = dp.get_array_length()
    
    vmin, vmax = value_range
    x_domain = (1, array_length)
    
    # Get control point positions and values
    xs, ys_float, xmin, xmax = _get_curve_data(side_key, array_length, x_domain, num_points)
    
    # Round control point y values to integers
    ys = np.round(ys_float).astype(int)
    ys = np.clip(ys, vmin, vmax)
    
    # Evaluate the smooth curve at each integer position
    lut_x = np.arange(1, array_length + 1)
    lut_y = _piecewise_hermite_smooth(xs, ys, lut_x)
    lut_y = np.clip(lut_y, vmin, vmax)
    
    # Round to integers for the LUT
    lut_y = np.round(lut_y).astype(int)
    
    return lut_y


def _get_curve_data(
    side_key: str = "sleeper_1",
    array_length: int = None,
    x_domain: Tuple[int, int] = None,
    num_points: int = 6,
    master_array: np.ndarray = None,
) -> tuple:
    """Get control points and interpolated curve data without rendering.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    array_length : int, optional
        Length of the master array. If None, retrieves from session state.
    x_domain : Tuple[int, int], optional
        Domain of x (min, max). If None, derived from array_length.
    num_points : int
        Number of control points
    master_array : np.ndarray, optional
        The master array to use. If None, fetches from session state.

    Returns
    -------
    np.ndarray
        The updated master array after applying any control changes.
    
    Returns
    -------
    tuple
        (xs, ys, xmin, xmax) - control point positions and values
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
    
    # Default control x positions (evenly spaced across domain)
    xs = np.linspace(xmin, xmax, num_points)
    xs = np.round(xs).astype(int)
    
    # Get y values from master array at control point positions
    # Directly sample the array at the xs positions (no interpolation)
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
        Number of control points to display
    master_array : np.ndarray, optional
        The master array to use. If None, fetches from session state.
    """
    if array_length is None:
        array_length = dp.get_array_length()

    # Always fetch fresh master array from session state for sliders
    master_array = dp.get_master_array(side_key, array_length=array_length)

    xs, ys_float, _, _ = _get_curve_data(side_key, array_length, x_domain, num_points, master_array)
    vmin, vmax = value_range
    
    # Round y values to integers for display
    ys = np.round(ys_float).astype(int)
    ys = np.clip(ys, vmin, vmax)
    
    st.subheader("Firmness Tuning")

    # Get base firmness to calculate offsets
    base_firmness = st.session_state.answers.get(side_key, {}).get("firmness_value", 2)
    
    updated_master = master_array.copy()
    updated = False

    for i in range(num_points):
        x_pos = int(xs[i])
        current_value = int(updated_master[x_pos - 1])
        offset = current_value - base_firmness  # Calculate offset from base

        col_label, col_minus, col_plus = st.columns([2.5, 1, 1], gap="small")
        with col_label:
            st.markdown(f"**Point {i+1} · Row {x_pos}**")

        with col_minus:
            if st.button("−", key=f"{side_key}_decrease_{i}", use_container_width=True, type="secondary"):
                if current_value > vmin:
                    updated_master[x_pos - 1] = current_value - 1
                    current_value = int(updated_master[x_pos - 1])
                    offset = current_value - base_firmness
                    updated = True

        with col_plus:
            if st.button("\+", key=f"{side_key}_increase_{i}", use_container_width=True, type="secondary"):
                if current_value < vmax:
                    updated_master[x_pos - 1] = current_value + 1
                    current_value = int(updated_master[x_pos - 1])
                    offset = current_value - base_firmness
                    updated = True

    if updated:
        updated_master = np.clip(updated_master, vmin, vmax).astype(int)
        dp.set_master_array(side_key, updated_master)

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
) -> None:
    """Render the curve plot visualization.
    
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
        Number of control points to display
    master_array : np.ndarray, optional
        The master array to use. If None, fetches from session state.
    """
    xs, ys_float, xmin, xmax = _get_curve_data(side_key, array_length, x_domain, num_points, master_array)
    vmin, vmax = value_range
    
    # Round control point y values to integers to match the controls display
    ys = np.round(ys_float).astype(int)
    ys = np.clip(ys, vmin, vmax)
    
    # Use piecewise Hermite interpolation for smooth curve display
    plot_samples = 1024
    plot_x = np.linspace(xmin, xmax, plot_samples)
    plot_y = _piecewise_hermite_smooth(xs, ys, plot_x)
    plot_y = np.clip(plot_y, vmin, vmax)

    fig = go.Figure()
    
    # Add smooth curve
    fig.add_trace(go.Scatter(
        x=plot_x,
        y=plot_y,
        mode='lines',
        name='Curve',
        line=dict(color='#0492a8', width=2)
    ))
    
    # Add control points
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='markers',
        name='Control Points',
        marker=dict(color='#0492a8', size=10)
    ))
    
    # Add labels above each control point
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(
            x=x,
            y=y,
            text=f"Point {i+1}",
            showarrow=False,
            yshift=15,
            font=dict(size=12, color='#6e6f72'),
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(4, 146, 168, 0)',
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
