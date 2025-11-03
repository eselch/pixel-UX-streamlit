"""Curve editor widget for Streamlit apps.

Provides show_curve_editor() which renders a cubic curve
through four control points. Points are adjustable with integer steps and
the resulting 256-sample lookup table (LUT) is saved in st.session_state.

Usage (from any page):
    from curve_editor import show_curve_editor
    lut = show_curve_editor(key="mycurve")

Returns:
    A list of 256 integers (0-255) representing the sampled curve values.
"""
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st


def _clamp_values(arr: np.ndarray, vmin: int = 0, vmax: int = 255) -> np.ndarray:
    return np.clip(np.round(arr).astype(int), vmin, vmax)


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
    key: str = "curve",
    initial: Optional[Tuple[int, int, int, int]] = None,
    x_domain: Tuple[int, int] = (0, 18),
    sample_count: Optional[int] = None,
    value_range: Tuple[int, int] = (1, 5),
    width: int = 700,
    height: int = 320,
    num_points: int = 6,
) -> List[int]:
    """Render a curve editor and return the sampled LUT.

    Parameters
    ----------
    key: session key prefix used for session_state storage
    initial: optional tuple of 4 initial y-values (integers). If None, defaults to flat 0..255 mapping.
    x_domain: domain of x (min,max) used for control points and sampling
    sample_count: how many samples to produce (usually 256)
    value_range: allowed y value range (min,max)
    width/height: size of the matplotlib figure

    Returns
    -------
    list of ints: sampled values length sample_count saved to st.session_state[f"{key}_lut"]
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for Snowflake/server environments
    import matplotlib.pyplot as plt
    
    xmin, xmax = x_domain
    vmin, vmax = value_range

    # If sample_count not provided, sample at integer x positions across domain
    if sample_count is None:
        sample_count = int(xmax - xmin + 1)

    # default control x positions (evenly spaced across domain) and snap to integers
    xs = np.linspace(xmin, xmax, num_points)
    xs = np.round(xs).astype(int)

    if initial is None:
        # default to the integer midpoint of the value range (e.g., 3 for 1..5)
        center = int(round((vmin + vmax) / 2.0))
        ys0 = np.full(len(xs), center, dtype=int)
    else:
        # allow a scalar (broadcast) or an iterable matching num_points
        if isinstance(initial, (int, float)):
            ys0 = np.full(len(xs), int(initial), dtype=int)
        else:
            try:
                init_list = list(initial)
            except TypeError:
                raise TypeError("initial must be an int/float or an iterable of ints")
            if len(init_list) == 1:
                ys0 = np.full(len(xs), int(init_list[0]), dtype=int)
            elif len(init_list) != len(xs):
                raise ValueError(f"initial must have length {len(xs)} (num_points) or be a scalar")
            else:
                ys0 = np.array([int(v) for v in init_list], dtype=int)

    # session state keys for the control points
    points_key = f"{key}_points"
    lut_key = f"{key}_lut"

    if points_key not in st.session_state:
        st.session_state[points_key] = ys0.tolist()

    # Render UI: two columns, left plot right controls
    col_plot, col_controls = st.columns([2, 1])

    with col_controls:
        st.subheader("Firmness Selector")
        # show four number inputs, step=1
        new_points = []
        for i in range(num_points):
            label = f"Point {i+1} (Row {int(xs[i])})"
            cur = st.number_input(
                label,
                min_value=int(vmin),
                max_value=int(vmax),
                value=int(st.session_state[points_key][i]),
                step=1,
                key=f"{points_key}_{i}",
            )
            new_points.append(int(cur))

        # update session state if changed
        if new_points != st.session_state[points_key]:
            st.session_state[points_key] = new_points

        st.markdown("---")
        st.caption("Adjust points in integer steps; the smooth cubic curve will update.")

    # Now compute cubic polynomial through the four control points
    # ensure control values are clamped to the allowed range
    ys = np.array(st.session_state[points_key], dtype=float)
    ys = np.clip(ys, vmin, vmax)
    st.session_state[points_key] = [int(v) for v in ys]

    # Use piecewise Hermite interpolation with zero derivatives at control
    # points so the handles are horizontal. This produces a smooth curve for
    # plotting while control points remain integers.
    # For display: sample a dense set of floating-point points so the plotted
    # curve looks smooth. We clip to the allowed value range but DO NOT round
    # values for plotting — rounding is only for the LUT/storage.
    plot_samples = 1024
    plot_x = np.linspace(xmin, xmax, plot_samples)
    plot_y = _piecewise_hermite_zero_deriv(xs, ys, plot_x)
    plot_y = np.clip(plot_y, vmin, vmax)

    # For LUT/storage: sample at the requested resolution (sample_count) and
    # round/clamp to integers before saving to session state.
    x_samples = np.linspace(xmin, xmax, sample_count)
    y_samples = _piecewise_hermite_zero_deriv(xs, ys, x_samples)
    y_samples = _clamp_values(y_samples, vmin, vmax)

    # Save LUT into session state
    st.session_state[lut_key] = y_samples.tolist()

    # Plot the curve and control points
    with col_plot:
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        # plot smooth curve from dense samples
        ax.plot(plot_x, plot_y, color="#0492a8", linewidth=2)
        # control points use primary teal color
        ax.scatter(xs, ys, color="#0492a8", s=80, zorder=5)
        # axis limits and ticks — Y limited to integer ticks between vmin..vmax
        # expand x-limits by 0.5 so half-integer centered labels are visible
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_ylim(vmin, vmax)
        ax.set_yticks(np.arange(int(vmin), int(vmax) + 1, 1))

        label_names = [str(int(t)) for t in range(int(xmin), int(xmax) + 1)]
        
        ax.set_xticks(np.linspace(int(xmin), int(xmax), num=len(label_names)))
        ax.set_xticklabels(label_names)
        ax.set_xlabel("Rows")
        ax.set_ylabel("Firmness Level")
        ax.set_title("Firmness Profile")
        ax.grid(axis='y', which='major', alpha=0.35)
        ax.grid(axis='x', which='minor', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    return st.session_state[lut_key]


if __name__ == "__main__":
    # simple local test when running this module directly (not necessary in app)
    if not st._is_running_with_streamlit:
        print("Run inside streamlit: streamlit run <app>")
