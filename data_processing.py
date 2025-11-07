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
    "Queen": (17, 6),
    "King": (17, 8),
    "St George King": (26, 13),
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


def draw_pixel_map(
    pixel_map_2d: np.ndarray,
    colorscale: list = None,
    show_values: bool = True,
    height: int = None,
    width: int = None,
) -> None:
    """Render a 2D firmness pixel map using Plotly heatmap.
    
    Parameters
    ----------
    pixel_map_2d : np.ndarray
        2D array (rows x cols) with integer values 0-4
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
    
    rows, cols = pixel_map_2d.shape
    
    # Calculate dimensions to force square cells
    cell_size = 30  # pixels per cell
    plot_width = cols * cell_size + 40  # Add margin
    plot_height = rows * cell_size + 40  # Add margin
    
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pixel_map_2d,
            x=list(range(cols)),  # 0-indexed x positions
            y=list(range(1, rows + 1)),  # 1-indexed y positions (1, 2, 3, ...)
            colorscale=colorscale,
            zmin=0,
            zmax=4,
            showscale=False,  # Hide colorscale bar
            text=pixel_map_2d if show_values else None,
            texttemplate="%{text}" if show_values else None,
            textfont=dict(size=12, color="white"),
            hoverinfo="none",  # Disable hover
            xgap=2,  # Add white gap between cells horizontally
            ygap=2,  # Add white gap between cells vertically
        )
    )
    
    # Update layout with square aspect ratio
    fig.update_layout(
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
        width=plot_width if width is None else width,
        margin=dict(l=20, r=5, t=5, b=5),  # Increased left margin for labels
        plot_bgcolor="white",
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=False, config={'staticPlot': True})


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