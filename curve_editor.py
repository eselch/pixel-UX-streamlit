"""Curve editor widget for Streamlit apps.

Provides show_curve_editor() which renders a cubic curve
through the master 1D array (domain 1-17) stored in session state via data_processing.

Usage (from any page):
    from curve_editor import show_curve_editor
    show_curve_editor(side_key="sleeper_1")

The function reads/writes to:
    st.session_state.answers[side_key]["master_array"]
"""
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go

import data_processing as dp


def _piecewise_hermite_zero_deriv(xs: np.ndarray, ys: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Piecewise cubic Hermite interpolation with zero derivatives at each control point.

    xs: 1D increasing x positions (length n)
    ys: 1D y values (length n)
    x_eval: points to evaluate (1D)

    Returns y values at x_eval.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    x_eval = np.asarray(x_eval)

    # prepare output
    y_out = np.empty_like(x_eval, dtype=float)

    # for each evaluation point, find segment
    # handle values <= xs[0] and >= xs[-1]
    left_mask = x_eval <= xs[0]
    right_mask = x_eval >= xs[-1]
    mid_mask = ~(left_mask | right_mask)

    y_out[left_mask] = ys[0]
    y_out[right_mask] = ys[-1]

    # indices for mid points
    if np.any(mid_mask):
        xm = x_eval[mid_mask]
        # find right index such that xs[i] <= x < xs[i+1]
        idx = np.searchsorted(xs, xm, side='right') - 1
        # clamp
        idx = np.clip(idx, 0, len(xs) - 2)

        x0 = xs[idx]
        x1 = xs[idx + 1]
        y0 = ys[idx]
        y1 = ys[idx + 1]
        t = (xm - x0) / (x1 - x0)
        # Hermite basis with zero derivatives reduces to simple blend
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = -2 * t ** 3 + 3 * t ** 2
        yvals = h00 * y0 + h10 * y1
        y_out[mid_mask] = yvals

    return y_out


def show_curve_editor(
    side_key: str = "sleeper_1",
    x_domain: Tuple[int, int] = (1, 17),
    value_range: Tuple[int, int] = (0, 4),
    width: int = 700,
    height: int = 320,
    num_points: int = 6,
) -> None:
    """Render a curve editor driven by the master array in data_processing.

    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    x_domain : Tuple[int, int]
        Domain of x (min, max) - typically (1, 17) matching master array
    value_range : Tuple[int, int]
        Allowed y value range (min, max)
    width/height : int
        Size of the plotly figure
    num_points : int
        Number of control points to display
    """
    xmin, xmax = x_domain
    vmin, vmax = value_range
    
    # Initialize master arrays if not already done
    dp.initialize_sleeper_master_arrays()
    
    # Get the master array for this sleeper
    master_array = dp.get_master_array(side_key)
    
    # Default control x positions (evenly spaced across domain)
    xs = np.linspace(xmin, xmax, num_points)
    xs = np.round(xs).astype(int)
    
    # Get y values from master array at control point x positions
    # Interpolate if x positions don't fall on integer indices
    ys = np.interp(xs, np.arange(1, 18), master_array)
    
    # Render UI: two columns, left plot right controls
    col_plot, col_controls = st.columns([3, 1])

    with col_controls:
        st.subheader("Firmness Selector")
        new_points = []
        
        for i in range(num_points):
            label = f"Point {i+1} (Row {int(xs[i])})"
            cur = st.number_input(
                label,
                min_value=int(vmin),
                max_value=int(vmax),
                value=int(ys[i]),
                step=1,
                key=f"{side_key}_point_{i}",
            )
            new_points.append(int(cur))
        
        # Update master array if control points changed
        if not np.array_equal(ys, new_points):
            # Interpolate full master array from updated control points
            updated_master = np.interp(np.arange(1, 18), xs, new_points)
            updated_master = np.clip(updated_master, vmin, vmax).astype(int)
            dp.set_master_array(side_key, updated_master)
            ys = new_points
        
        st.markdown("---")
        st.caption("Adjust points to customize firmness profile.")

    # Use piecewise Hermite interpolation for smooth curve display
    plot_samples = 1024
    plot_x = np.linspace(xmin, xmax, plot_samples)
    plot_y = _piecewise_hermite_zero_deriv(xs, ys, plot_x)
    plot_y = np.clip(plot_y, vmin, vmax)

    # Plot the curve and control points using Plotly
    with col_plot:
        st.subheader("Firmness Profile")

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
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
        st.write("")


if __name__ == "__main__":
    # simple local test when running this module directly (not necessary in app)
    if not st._is_running_with_streamlit:
        print("Run inside streamlit: streamlit run <app>")
