"""Data processing utilities for the Xsensor survey application.

Manages the master 1D array for each sleeper that drives the curve editor
and other downstream visualizations.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Master array configuration
MASTER_VALUE_RANGE = (0, 4)  # Value range for array elements

# Firmness options and labels
FIRMNESS_OPTIONS = [0, 1, 2, 3, 4]
FIRMNESS_LABELS = ["Very Soft (0)", "Soft (1)", "Medium (2)", "Firm (3)", "Very Firm (4)"]

# Bed size configurations: name -> (length, width)
# Single source of truth for array dimensions
BED_SIZES = {
    "Queen": (17, 12),
    "King": (17, 16),
    "St George King": (26, 26),
}


def set_bed_size(bed_name: str) -> None:
    """Set the bed size and reinitialize master arrays fresh at new dimensions.
    
    When bed size changes:
    1. Store new bed size in session state
    2. Discard old master arrays (don't interpolate)
    3. Create fresh arrays at new size with current firmness values
    4. User edits/transformations will be applied after resize
    
    Parameters
    ----------
    bed_name : str
        Bed size name (must be in BED_SIZES dict)
    """
    if bed_name not in BED_SIZES:
        st.error(f"Unknown bed size: {bed_name}")
        return
    
    # Store new bed size
    st.session_state["bed_size"] = bed_name
    new_length, new_width = BED_SIZES[bed_name]
    
    # Reinitialize all sleeper master arrays fresh at new size
    # This discards old arrays and creates new ones
    if "answers" in st.session_state:
        for side_key in ["sleeper_1", "sleeper_2"]:
            if side_key in st.session_state.answers:
                # Get the firmness value to maintain user's preference
                firmness = st.session_state.answers[side_key].get("firmness_value", 2)
                # Create fresh array at new size (overwrites old array)
                fresh_array = initialize_master_array(side_key, firmness, array_length=new_length)
                st.session_state.answers[side_key]["master_array"] = fresh_array.tolist()


def get_bed_size() -> tuple:
    """Get the current bed size (length, width) from session state.
    
    Returns
    -------
    tuple
        (length, width) for the selected bed size, defaults to Queen if not set
    """
    bed_name = st.session_state.get("bed_size", "Queen")
    return BED_SIZES[bed_name]


def get_array_length() -> int:
    """Get the current array length from the selected bed size.
    
    Returns
    -------
    int
    
    """
    length, _ = get_bed_size()
    return length


def get_array_width() -> int:
    """Get the current array width from the selected bed size.
    
    Returns
    -------
    int
        The array width (bed width), defaults to Queen (6) if no bed size set
    """
    _, width = get_bed_size()
    return width


def pixel_map(master_array: np.ndarray, width: int) -> np.ndarray:
    """Convert a 1D master array into a 2D pixel map by duplicating it across columns.
    
    Takes a 1D array and replicates it horizontally to create a 2D array where
    each row is identical to the master array.
    
    Parameters
    ----------
    master_array : np.ndarray
        1D array of length array_length
    width : int
        Number of columns (width) for the output 2D array
    
    Returns
    -------
    np.ndarray
        2D array of shape (len(master_array), width) where each row is the master_array
    """
    # Ensure master_array is a numpy array
    master_array = np.asarray(master_array, dtype=int)
    
    # Replicate the 1D array across columns to create 2D array
    pixel_map_2d = np.tile(master_array[:, np.newaxis], (1, width))
    
    return pixel_map_2d


def pixel_map_dual_sleeper(
    left_array: np.ndarray, 
    right_array: np.ndarray, 
    width_per_sleeper: int
) -> np.ndarray:
    """Create a 2D pixel map with two sleeper arrays side-by-side.
    
    The left half of the heatmap shows left_array data, and the right half
    shows right_array data. Each sleeper's 1D array is replicated across
    their respective columns.
    
    Parameters
    ----------
    left_array : np.ndarray
        1D array for the left sleeper (length = array_length)
    right_array : np.ndarray
        1D array for the right sleeper (length = array_length)
    width_per_sleeper : int
        Number of columns for each sleeper's half
    
    Returns
    -------
    np.ndarray
        2D array of shape (array_length, width_per_sleeper * 2)
        Left half uses left_array, right half uses right_array
    """
    # Ensure arrays are numpy arrays
    left_array = np.asarray(left_array, dtype=int)
    right_array = np.asarray(right_array, dtype=int)
    
    # Create pixel maps for each sleeper
    left_map = np.tile(left_array[:, np.newaxis], (1, width_per_sleeper))
    right_map = np.tile(right_array[:, np.newaxis], (1, width_per_sleeper))
    
    # Concatenate horizontally (left | right)
    dual_map = np.hstack([left_map, right_map])
    
    return dual_map


def downsample_pressure_map(
    pressure_array: np.ndarray,
    target_shape: tuple = (17, 9)
) -> np.ndarray:
    """Downsample a high-resolution pressure map to a lower resolution grid.
    
    Handles automatic rotation if the array comes in rotated (64x160 instead of 160x64).
    Uses block averaging to preserve spatial information while reducing resolution.
    
    Parameters
    ----------
    pressure_array : np.ndarray
        Input pressure map, expected to be 160x64 or 64x160
    target_shape : tuple
        Target dimensions (rows, cols), default is (17, 9) for Queen bed
    
    Returns
    -------
    np.ndarray
        Downsampled array of shape target_shape
        
    Notes
    -----
    - If input is 64x160, it will be rotated 90° counterclockwise to 160x64
    - Uses block averaging: divides array into blocks and takes mean of each block
    - Preserves spatial layout and pressure distribution
    """
    pressure_array = np.asarray(pressure_array, dtype=float)
    
    # Check if array needs rotation (64x160 instead of 160x64)
    rows, cols = pressure_array.shape
    if rows == 64 and cols == 160:
        # Rotate 90° counterclockwise: rot90(k=1)
        pressure_array = np.rot90(pressure_array, k=1)
        rows, cols = pressure_array.shape
    
    # Verify we have the expected shape (160x64)
    if rows != 160 or cols != 64:
        raise ValueError(
            f"Expected pressure map to be 160x64 or 64x160, got {pressure_array.shape}. "
            "Please check the input data."
        )
    
    target_rows, target_cols = target_shape
    
    # Calculate block sizes for downsampling
    # Each block in the output represents a region in the input
    block_height = rows / target_rows  # e.g., 160 / 17 ≈ 9.41
    block_width = cols / target_cols    # e.g., 64 / 9 ≈ 7.11
    
    # Create output array
    downsampled = np.zeros(target_shape, dtype=float)
    
    # Downsample using block averaging
    for i in range(target_rows):
        for j in range(target_cols):
            # Calculate block boundaries in original array
            row_start = int(i * block_height)
            row_end = int((i + 1) * block_height)
            col_start = int(j * block_width)
            col_end = int((j + 1) * block_width)
            
            # Extract block and compute mean
            block = pressure_array[row_start:row_end, col_start:col_end]
            downsampled[i, j] = np.mean(block)
    
    return downsampled


def pressure_map_to_1d_array(pressure_map_2d: np.ndarray) -> np.ndarray:
    """Convert a 2D pressure map to a 1D array using the max value from each row.
    
    Takes a 2D array (typically 17x9 for a Queen bed) and creates a 1D array
    where each element is the maximum value from the corresponding row.
    
    Parameters
    ----------
    pressure_map_2d : np.ndarray
        2D pressure map array, typically shape (17, 9)
    
    Returns
    -------
    np.ndarray
        1D array with length equal to number of rows, where each element
        is the max value from that row
        
    Examples
    --------
    >>> pressure_map = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> pressure_map_to_1d_array(pressure_map)
    array([3, 6, 9])
    """
    pressure_map_2d = np.asarray(pressure_map_2d)
    
    # Get max value from each row (axis=1)
    return np.max(pressure_map_2d, axis=1)


def apply_pressure_map_to_curve(
    pressure_map_2d: np.ndarray,
    side_key: str = "sleeper_1",
    num_control_points: int = 6,
    remap_range: str = "low"
) -> None:
    """Apply pressure map data using scipy spline fitting.
    
    Creates a scipy UnivariateSpline from the pressure data and stores
    the spline's knot positions and values for editing in spline mode.
    
    This function:
    1. Converts the 2D pressure map to a 1D array (max per row)
    2. Normalizes and inverts the data
    3. Fits a scipy spline to the data
    4. Extracts the spline's internal knots as control points
    5. Stores the knots and enables "spline mode" in the curve editor
    
    Parameters
    ----------
    pressure_map_2d : np.ndarray
        2D pressure map array, typically shape (17, 9)
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    num_control_points : int
        Unused in spline mode (knot count is determined by scipy)
    remap_range : str
        Remap range: "extra_low" (1-2), "low" (1-3), or "high" (0-4), default "low"
    
    Notes
    -----
    The master array is updated with the scipy spline fit.
    High pressure areas become soft (low values), low pressure areas become firm (high values).
    The spline's internal knots become the editable control points.
    """
    # Convert 2D pressure map to 1D array (max per row)
    pressure_1d = pressure_map_to_1d_array(pressure_map_2d)
    array_length = len(pressure_1d)
    
    # Determine remap range
    if remap_range == "extra_low":
        range_min, range_max = 1, 2
    elif remap_range == "low":
        range_min, range_max = 1, 3
    else:  # "high"
        range_min, range_max = 0, 4
    
    # Normalize pressure_1d to the specified range
    p_min = np.min(pressure_1d)
    p_max = np.max(pressure_1d)
    
    if p_max > p_min:
        # Scale to specified range
        normalized = range_min + (range_max - range_min) * (pressure_1d - p_min) / (p_max - p_min)
    else:
        # All values are the same, set to middle value
        mid_value = (range_min + range_max) / 2.0
        normalized = np.full_like(pressure_1d, mid_value, dtype=float)
    
    # INVERT: high pressure → soft (low value), low pressure → firm (high value)
    inverted = range_max + range_min - normalized
    
    # === FIT SCIPY SPLINE ===
    try:
        from scipy.interpolate import UnivariateSpline
        
        # Create x data (0-indexed positions)
        x_data = np.arange(array_length)
        
        # Fit a smoothing spline to the inverted pressure data
        # s = smoothing factor: 0 = interpolate exactly, higher = smoother
        # Initialize at 5% of array length
        smoothing_factor = array_length * 0.05
        
        # Use cubic spline (k=3) for smooth curves
        spline = UnivariateSpline(x_data, inverted, s=smoothing_factor, k=3)
        
        # Get the spline's internal knots (these become the control points)
        # UnivariateSpline internal knots exclude the boundary knots
        knots_internal = spline.get_knots()
        
        # Ensure we have at least 4 knots for good control
        min_knots = 4
        if len(knots_internal) < min_knots:
            # Use exact interpolation through more points
            # Select evenly-spaced positions as knots
            num_knots = max(min_knots, array_length // 3)  # At least 4, or 1 per 3 rows
            knot_indices = np.linspace(0, array_length - 1, num_knots)
            knot_indices = np.round(knot_indices).astype(int)
            
            # Fit spline with exact interpolation (s=0) through selected points
            spline = UnivariateSpline(knot_indices, inverted[knot_indices], s=0, k=3)
            knots_internal = knot_indices.astype(float)
        
        # Evaluate spline at knot positions to get Y values
        knot_y_values = spline(knots_internal)
        
        # Convert knots to 1-indexed for consistency with row numbers
        knots_1indexed = np.round(knots_internal + 1).astype(int)
        
        # Clip knot positions to valid range
        knots_1indexed = np.clip(knots_1indexed, 1, array_length)
        
        # Clip Y values to the remap range
        knot_y_values = np.clip(knot_y_values, range_min, range_max)
        
        # Round knot Y values to integers (they'll be rounded anyway in master array)
        knot_y_values = np.round(knot_y_values).astype(float)
        
        # Refit spline with rounded knot values for exact interpolation
        # This ensures the curve passes through integer firmness levels at knots
        xs_0indexed = knots_1indexed - 1
        spline = UnivariateSpline(xs_0indexed, knot_y_values, s=0, k=min(3, len(knots_1indexed) - 1))
        
        # Evaluate the spline across the entire array to create master_array
        x_full = np.arange(array_length)
        master_array_values = spline(x_full)
        master_array_values = np.clip(master_array_values, range_min, range_max)
        master_array_values = np.round(master_array_values).astype(int)
        
        # === STORE SPLINE DATA IN SESSION STATE ===
        if "answers" not in st.session_state:
            st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}
        
        if side_key not in st.session_state.answers:
            st.session_state.answers[side_key] = {}
        
        # Enable spline mode
        st.session_state.answers[side_key]["use_scipy_spline"] = True
        
        # Store the ORIGINAL normalized and inverted 1D pressure data
        # This is the source data we'll always refit from
        st.session_state.answers[side_key]["original_pressure_1d"] = inverted.tolist()
        
        # Store knot positions and values (these are the editable control points)
        st.session_state.answers[side_key]["spline_knots"] = {
            "x": knots_1indexed.tolist(),
            "y": knot_y_values.tolist()
        }
        
        # Store the smoothing factor for reference
        st.session_state.answers[side_key]["spline_smoothing"] = smoothing_factor
        
        # Store the remap range
        st.session_state.answers[side_key]["remap_range"] = remap_range
        
        # Store the initial curve scale as 100% (fills full graph range)
        st.session_state.answers[side_key]["curve_scale_percent"] = 100.0
        
        # Store the original firmness at time of upload (for calculating offsets later)
        original_firmness = st.session_state.answers[side_key].get("firmness_value", 2)
        st.session_state.answers[side_key]["original_firmness"] = original_firmness
        
        # Store master array
        st.session_state.answers[side_key]["master_array"] = master_array_values.tolist()
        
        # Store the number of knots (for UI to know how many controls to show)
        st.session_state.answers[side_key]["num_control_points"] = len(knots_1indexed)
        
    except Exception as e:
        # If spline fitting fails, fall back to simple normalized data
        master_array = np.round(inverted).astype(int)
        master_array = np.clip(master_array, range_min, range_max)
        set_master_array(side_key, master_array)
        pass


def refit_spline_from_original(
    side_key: str = "sleeper_1",
    scale_percent: float = 100.0,
    firmness_offset: float = 0.0,
    smoothing_factor: float = None
) -> None:
    """Refit the spline from the original pressure data with scale and firmness adjustments.
    
    This allows adjusting the curve without losing data to clipping. The original
    pressure data is scaled and shifted, then the spline is refit to produce new knots.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    scale_percent : float
        Scale percentage (1-100%). 100% = data fills full 0-4 range, 1% = nearly flat line
    firmness_offset : float
        Offset to add after scaling (shifts the whole curve up/down)
    smoothing_factor : float, optional
        Smoothing factor for scipy spline. If None, uses stored value.
    
    Notes
    -----
    Reads from st.session_state.answers[side_key]["original_pressure_1d"]
    Updates spline_knots and master_array in session state.
    """
    if "answers" not in st.session_state:
        return
    
    if side_key not in st.session_state.answers:
        return
    
    sleeper_data = st.session_state.answers[side_key]
    
    # Get the original pressure data
    original_data = sleeper_data.get("original_pressure_1d")
    if original_data is None:
        return
    
    original_data = np.array(original_data, dtype=float)
    array_length = len(original_data)
    
    # Get smoothing factor (use provided value, stored value, or default 5%)
    if smoothing_factor is None:
        smoothing_factor = sleeper_data.get("spline_smoothing", array_length * 0.05)
    else:
        # Store the new smoothing factor
        sleeper_data["spline_smoothing"] = smoothing_factor
    
    # Calculate the original data range
    data_min = np.min(original_data)
    data_max = np.max(original_data)
    data_range = data_max - data_min
    data_center = (data_max + data_min) / 2.0
    
    # Convert percentage to scale factor
    # At 100%, the data should fill the full 0-4 range (scale = 4.0 / data_range)
    # At 1%, the data should be nearly flat (scale approaches 0)
    full_range = 4.0
    if data_range > 0:
        # Scale factor to make data fill full range at 100%
        scale_for_full_range = full_range / data_range
        # Apply percentage: 100% = full scale, 1% = 1% of full scale
        scale_factor = scale_for_full_range * (scale_percent / 100.0)
    else:
        scale_factor = 1.0
    
    # Apply scale around center point
    scaled_data = data_center + (original_data - data_center) * scale_factor
    
    # Apply firmness offset
    adjusted_data = scaled_data + firmness_offset
    
    # Clip to valid range [0, 4]
    adjusted_data = np.clip(adjusted_data, 0, 4)
    
    # === FIT SCIPY SPLINE ===
    try:
        from scipy.interpolate import UnivariateSpline
        
        # Create x data (0-indexed positions)
        x_data = np.arange(array_length)
        
        # Fit a smoothing spline to the adjusted data
        spline = UnivariateSpline(x_data, adjusted_data, s=smoothing_factor, k=3)
        
        # Get the spline's internal knots
        knots_internal = spline.get_knots()
        
        # Ensure we have at least 4 knots for good control
        min_knots = 4
        if len(knots_internal) < min_knots:
            # Use exact interpolation through more points
            num_knots = max(min_knots, array_length // 3)
            knot_indices = np.linspace(0, array_length - 1, num_knots)
            knot_indices = np.round(knot_indices).astype(int)
            
            # Fit spline with exact interpolation (s=0) through selected points
            spline = UnivariateSpline(knot_indices, adjusted_data[knot_indices], s=0, k=3)
            knots_internal = knot_indices.astype(float)
        
        # Evaluate spline at knot positions to get Y values
        knot_y_values = spline(knots_internal)
        
        # Convert knots to 1-indexed for consistency with row numbers
        knots_1indexed = np.round(knots_internal + 1).astype(int)
        knots_1indexed = np.clip(knots_1indexed, 1, array_length)
        
        # Clip Y values to valid range
        knot_y_values = np.clip(knot_y_values, 0, 4)
        
        # Round knot Y values to integers (they'll be rounded anyway in master array)
        knot_y_values = np.round(knot_y_values).astype(float)
        
        # Refit spline with rounded knot values for exact interpolation
        # This ensures the curve passes through integer firmness levels at knots
        xs_0indexed = knots_1indexed - 1
        spline = UnivariateSpline(xs_0indexed, knot_y_values, s=0, k=min(3, len(knots_1indexed) - 1))
        
        # Evaluate the spline across the entire array to create master_array
        x_full = np.arange(array_length)
        master_array_values = spline(x_full)
        master_array_values = np.clip(master_array_values, 0, 4)
        master_array_values = np.round(master_array_values).astype(int)
        
        # Update stored knots
        sleeper_data["spline_knots"] = {
            "x": knots_1indexed.tolist(),
            "y": knot_y_values.tolist()
        }
        
        # Update number of control points (may change with refitting)
        sleeper_data["num_control_points"] = len(knots_1indexed)
        
        # Update master array
        set_master_array(side_key, master_array_values)
        
    except Exception as e:
        print(f"Error refitting spline from original data: {e}")


def draw_pixel_map(
    pixel_map_2d: np.ndarray,
    colorscale: list = None,
    show_values: bool = True,
    height: int = None,
    width: int = None,
    value_range: tuple = None,
    use_container_width: bool = False,
    title: str = None,
) -> None:
    """Render a 2D firmness pixel map using Plotly heatmap.
    
    Parameters
    ----------
    pixel_map_2d : np.ndarray
        2D array (rows x cols) with values
    colorscale : list, optional
        Custom colorscale as [[position, color], ...] where position is 0.0-1.0
        and color is hex/rgb/rgba. If None, uses default blue gradient.
        Example: [[0.0, "#e3f2fd"], [0.25, "#90caf9"], [0.5, "#42a5f5"], 
                  [0.75, "#1e88e5"], [1.0, "#0d47a1"]]
    show_values : bool
        Whether to display firmness values as text in each cell
    height : int, optional
        Chart height in pixels. If None, auto-calculated from array size.
    width : int, optional
        Chart width in pixels. If None, auto-calculated from array size.
    value_range : tuple, optional
        Value range for the colorscale. Can be:
        - (min, max): explicit range like (0, 4) or (1, 100)
        - "auto": automatically uses (data.min(), data.max())
        - None: defaults to (0, 4) for backward compatibility
    title : str, optional
        Title to display above the heatmap
    """
    # Default colorscale if none provided (blue gradient for firmness levels 0-4)
    if colorscale is None:
        colorscale = [
            [0.0, "#e3f2fd"],    # 0 - Very Soft (light blue)
            [0.25, "#90caf9"],   # 1 - Soft (medium blue)
            [0.5, "#42a5f5"],    # 2 - Medium (blue)
            [0.75, "#1e88e5"],   # 3 - Firm (dark blue)
            [1.0, "#0d47a1"],    # 4 - Very Firm (darkest blue)
        ]
    
    # Determine value range for colorscale
    if value_range == "auto":
        zmin = float(np.min(pixel_map_2d))
        zmax = float(np.max(pixel_map_2d))
    elif value_range is not None:
        zmin, zmax = value_range
    else:
        # Default to 0-4 for backward compatibility
        zmin, zmax = 0, 4
    
    rows, cols = pixel_map_2d.shape
    
    # Calculate dimensions to force square cells
    cell_size = 30  # pixels per cell
    plot_width = cols * cell_size + 40  # Add margin
    plot_height = rows * cell_size + 40  # Add margin
    
    # If width is specified, calculate proportional height for square cells
    if width is not None:
        # Calculate cell size based on fixed width
        actual_cell_size = (width - 40) / cols
        plot_height = rows * actual_cell_size + 40
        plot_width = width

    # If using container width, let Streamlit manage figure width
    fig_width = plot_width if width is None else width
    if use_container_width:
        fig_width = None
    
    pixel_map_2d_rounded = np.round(pixel_map_2d, decimals=2, out=pixel_map_2d)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pixel_map_2d,
            x=list(range(cols)),  # 0-indexed x positions
            y=list(range(1, rows + 1)),  # 1-indexed y positions (1, 2, 3, ...)
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=False,  # Hide colorscale bar
            text=pixel_map_2d_rounded if show_values else None,
            texttemplate="%{text}" if show_values else None,
            textfont=dict(size=12, color="white"),
            hoverinfo="none",  # Disable hover
            xgap=2,  # Add white gap between cells horizontally
            ygap=2,  # Add white gap between cells vertically
        )
    )
    
    # Update layout with square aspect ratio
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ) if title else None,
        xaxis=dict(
            side="top",
            showgrid=False,
            showticklabels=False,  # Hide x-axis numbers
            ticks="",  # Remove tick marks
            showline=False,  # Hide x-axis line
            zeroline=False,  # Hide zero line
            range=[-0.5, cols - 0.5],
            constrain="domain",
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=True,  # Show y-axis numbers (row labels)
            ticks="",  # Remove tick marks
            tickmode='array',
            tickvals=list(range(1, rows+1)),  # Show all ticks (1 to rows)
            dtick=1,  # Show every row number
            showline=False,  # Hide y-axis line
            zeroline=False,  # Hide zero line
            autorange="reversed",  # Row 1 at top
            range=[0.5, rows + 0.5],  # Adjusted for 1-indexed y values
            scaleanchor="x",  # Force square cells
            scaleratio=1,
        ),
        height=plot_height if height is None else height,
        width=fig_width,
        margin=dict(l=20, r=5, t=40 if title else 5, b=5),  # Increased top margin for title
        plot_bgcolor="white",
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=use_container_width, config={'staticPlot': True})


def initialize_master_array(side_key: str, firmness_value: int = 2, array_length: int = None) -> np.ndarray:
    """Initialize a master 1D array for a sleeper.
    
    Creates a flat array initialized to the firmness value selected in the survey.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    firmness_value : int
        The initial firmness level (0-4) from the survey questions.
    array_length : int, optional
        Length of the array to create. If None, uses current bed size length.
    
    Returns
    -------
    np.ndarray
        1D array of length array_length with all values set to firmness_value (as int)
    """
    if array_length is None:
        array_length = get_array_length()
    
    # Clamp firmness value to valid range
    firmness_value = int(np.clip(firmness_value, MASTER_VALUE_RANGE[0], MASTER_VALUE_RANGE[1]))
    # Create array, initialized to firmness value
    master_array = np.full(array_length, firmness_value, dtype=int)
    return master_array


def initialize_sleeper_master_arrays(array_length: int = None):
    """Initialize master arrays in session state for all sleepers.
    
    Call this on pages where you need master array access.
    Checks if sleeper has survey data and initializes/updates the master array
    based on the firmness value from the survey.
    
    Parameters
    ----------
    array_length : int, optional
        Length of arrays to create. If None, uses current bed size length.
    """
    if array_length is None:
        array_length = get_array_length()
    
    if "answers" not in st.session_state:
        st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}
    
    # Initialize sleeper_1 master array
    if "sleeper_1" in st.session_state.answers:
        sleeper_1_data = st.session_state.answers["sleeper_1"]
        firmness_1 = sleeper_1_data.get("firmness_value", 2)  # default to 2 (middle of 0-4)
        
        if "master_array" not in sleeper_1_data:
            sleeper_1_data["master_array"] = initialize_master_array("sleeper_1", firmness_1, array_length).tolist()
        else:
            # Update master array if firmness changed
            if not isinstance(sleeper_1_data["master_array"], list):
                sleeper_1_data["master_array"] = sleeper_1_data["master_array"].tolist()
    
    # Initialize sleeper_2 master array if they've been added
    if st.session_state.get("show_right", False) and "sleeper_2" in st.session_state.answers:
        sleeper_2_data = st.session_state.answers["sleeper_2"]
        firmness_2 = sleeper_2_data.get("firmness_value", 2)  # default to 2 (middle of 0-4)
        
        if "master_array" not in sleeper_2_data:
            sleeper_2_data["master_array"] = initialize_master_array("sleeper_2", firmness_2, array_length).tolist()
        else:
            # Update master array if firmness changed
            if not isinstance(sleeper_2_data["master_array"], list):
                sleeper_2_data["master_array"] = sleeper_2_data["master_array"].tolist()


def get_master_array(side_key: str, array_length: int = None) -> np.ndarray:
    """Retrieve the master array for a sleeper as a numpy array.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    array_length : int, optional
        Expected length of the array. If None, uses current bed size length.
    
    Returns
    -------
    np.ndarray
        The master 1D array (dtype int) as a numpy array, or default if not found
    """
    if array_length is None:
        array_length = get_array_length()
    
    if "answers" in st.session_state and side_key in st.session_state.answers:
        sleeper_data = st.session_state.answers[side_key]
        if "master_array" in sleeper_data:
            return np.array(sleeper_data["master_array"], dtype=int)
    
    # Return default if not found
    return np.full(array_length, MASTER_VALUE_RANGE[0], dtype=int)


def set_master_array(side_key: str, array: np.ndarray) -> None:
    """Update the master array for a sleeper in session state.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    array : np.ndarray or list
        The new master array (length 17) to store
    """
    if "answers" not in st.session_state:
        st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}
    
    if side_key not in st.session_state.answers:
        st.session_state.answers[side_key] = {}
    
    # Convert to list for JSON serialization
    if isinstance(array, np.ndarray):
        st.session_state.answers[side_key]["master_array"] = array.tolist()
    else:
        st.session_state.answers[side_key]["master_array"] = list(array)


def update_master_array_from_firmness(side_key: str, firmness_value: int) -> None:
    """Update the master array when firmness value changes.

    Reinitializes all values to match the new firmness level.

    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    firmness_value : int
        The new firmness level (1-5)
    """
    new_array = initialize_master_array(side_key, firmness_value)
    set_master_array(side_key, new_array)