import streamlit as st
from datetime import datetime
import numpy as np
import common as ui
import data_processing as dp
from color_grid_component import render_color_grid

ui.apply_base_ui("Pixel Sandbox")

st.write("Click and drag to select pixels and change their firmness")

# Initialize session state if not already done
if "answers" not in st.session_state:
    st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}

# Initialize master arrays from data_processing
dp.initialize_sleeper_master_arrays()

# Initialize bed size configuration
st.session_state.setdefault("bed_size", "King")

# Get current bed dimensions
bed_length, bed_width = dp.get_bed_size()

# Initialize sleeper visibility
st.session_state.setdefault("show_right", False)

# Initialize sleeper side preference
st.session_state.setdefault("sleeper_1_on_left", True)

# Initialize current sleeper selection
st.session_state.setdefault("current_configure_sleeper", "Sleeper 1")

# Initialize CSV data storage
if "csv_data" not in st.session_state:
    st.session_state.csv_data = {}

# Initialize firmness values if not set
for side_key in ["sleeper_1", "sleeper_2"]:
    if side_key not in st.session_state.answers:
        st.session_state.answers[side_key] = {}
    st.session_state.answers[side_key].setdefault("firmness_value", 2)
    st.session_state.answers[side_key].setdefault("use_scipy_spline", False)

# Initialize selection state for sandbox grid
if "selected_cells" not in st.session_state:
    st.session_state.selected_cells = set()


def get_interpolated_curve_lut(side_key: str) -> np.ndarray:
    """Get the interpolated curve LUT for a sleeper, matching Configure page logic."""
    sleeper_data = st.session_state.answers.get(side_key, {})
    master_array = sleeper_data.get("master_array")
    if master_array is not None:
        return np.array(master_array, dtype=int)
    # Fallback to firmness-based default
    firmness = sleeper_data.get("firmness_value", 2)
    return dp.initialize_master_array(side_key, firmness, array_length=bed_length)


def build_pixel_map_from_configure() -> np.ndarray:
    """Build the pixel map 2D array using the same logic as Configure page."""
    show_right = st.session_state.get("show_right", False)
    current_sleeper = st.session_state.get("current_configure_sleeper", "Sleeper 1")
    side_key = "sleeper_1" if current_sleeper == "Sleeper 1" else "sleeper_2"

    total_width = dp.get_array_width()
    width_per_sleeper = total_width // 2

    if show_right:
        # Dual sleeper mode
        if st.session_state.get("sleeper_1_on_left", True):
            left_key = "sleeper_1"
            right_key = "sleeper_2"
        else:
            left_key = "sleeper_2"
            right_key = "sleeper_1"

        left_lut = get_interpolated_curve_lut(left_key)
        right_lut = get_interpolated_curve_lut(right_key)

        if left_lut is None or right_lut is None:
            return np.full((bed_length, total_width), 2, dtype=int)

        return dp.pixel_map_dual_sleeper(left_lut, right_lut, width_per_sleeper)
    else:
        # Single sleeper mode - same data on both sides
        curve_lut = get_interpolated_curve_lut(side_key)

        if curve_lut is None:
            return np.full((bed_length, total_width), 2, dtype=int)

        return dp.pixel_map_dual_sleeper(curve_lut, curve_lut, width_per_sleeper)


def create_pixel_assortment(width: int) -> np.ndarray:
    """Create a 5-row pixel assortment array with each row being a different firmness (0-4).
    
    Args:
        width: The width of the bed (number of columns)
    
    Returns:
        np.ndarray: A 5 x width array where row 0 is all 0s, row 1 is all 1s, etc.
    """
    assortment = np.zeros((5, width), dtype=int)
    for i in range(5):
        assortment[i, :] = i
    return assortment


def apply_pixel_assortment():
    """Apply the pixel assortment to the bottom 5 rows of the sandbox grid."""
    if "sandbox_pixel_grid_modified" not in st.session_state:
        st.session_state.sandbox_pixel_grid_modified = build_pixel_map_from_configure()
    
    grid = st.session_state.sandbox_pixel_grid_modified
    total_width = grid.shape[1]
    
    # Create the assortment
    assortment = create_pixel_assortment(total_width)
    
    # Overwrite the bottom 5 rows
    grid[-5:, :] = assortment
    
    st.session_state.sandbox_pixel_grid_modified = grid


def get_master_arrays_signature() -> str:
    """Generate a signature of the current master arrays to detect changes."""
    sleeper_1_array = st.session_state.answers.get("sleeper_1", {}).get("master_array", [])
    sleeper_2_array = st.session_state.answers.get("sleeper_2", {}).get("master_array", [])
    show_right = st.session_state.get("show_right", False)
    sleeper_1_on_left = st.session_state.get("sleeper_1_on_left", True)
    # Create a hashable signature from the arrays and configuration
    return f"{tuple(sleeper_1_array)}_{tuple(sleeper_2_array)}_{show_right}_{sleeper_1_on_left}"


# Track the bed size and master arrays to detect changes
current_bed_size = st.session_state.get("bed_size", "King")
current_master_signature = get_master_arrays_signature()

# Initialize or rebuild the sandbox grid if bed size or master arrays changed
if "sandbox_pixel_grid_modified" not in st.session_state:
    st.session_state.sandbox_pixel_grid_modified = build_pixel_map_from_configure()
    st.session_state.sandbox_last_bed_size = current_bed_size
    st.session_state.sandbox_last_master_signature = current_master_signature
elif (st.session_state.get("sandbox_last_bed_size") != current_bed_size or 
      st.session_state.get("sandbox_last_master_signature") != current_master_signature):
    # Bed size or master arrays changed - rebuild the grid
    st.session_state.sandbox_pixel_grid_modified = build_pixel_map_from_configure()
    st.session_state.sandbox_last_bed_size = current_bed_size
    st.session_state.sandbox_last_master_signature = current_master_signature

# Use the modified grid (persists user changes)
sandbox_pixel_grid = st.session_state.sandbox_pixel_grid_modified

# Firmness color palette matching Lovesac theme
firmness_colors = [
    "#E9F1F0",  # 0: Very Soft
    "#A7C7BF",  # 1: Soft
    "#1E99A8",  # 2: Medium
    "#006261",  # 3: Firm
    "#0A2734",  # 4: Very Firm
]
firmness_names = ["Very Soft (0)", "Soft (1)", "Medium (2)", "Firm (3)", "Very Firm (4)"]

# Get sleeper names for screenshot subtitle
sleeper_1_name = st.session_state.answers.get("sleeper_1", {}).get("setting1", "Sleeper 1")
sleeper_2_name = st.session_state.answers.get("sleeper_2", {}).get("setting1", "Sleeper 2")
screenshot_subtitle = f"{sleeper_1_name} and {sleeper_2_name}"

# Convert numpy array to list of lists for the component
initial_grid = sandbox_pixel_grid.tolist()

# Render the interactive color grid
render_color_grid(
    initial_pattern=initial_grid,
    colors=firmness_colors,
    color_names=firmness_names,
    cell_size=35,
    spacing=2,
    session_key="sandbox_pixel_grid_state",
    screenshot_subtitle=screenshot_subtitle,
    cell_border="1px solid #333",
    selected_border="2px solid #FFD54F",
    selected_shadow="0 0 0 2px #FFD54F",
)

# Bottom navigation buttons
st.markdown("---")

col_prev, col_spacer = st.columns([1, 3])

with col_prev:
    if st.button("← Previous", width="content", type="secondary"):
        st.switch_page("pages/3_Configure.py")
