"""CSV loader module for importing Xsensor pressure map data.

Handles single or multiple CSV files, extracts statistics, and merges sensel data.
Adapted for Streamlit file upload interface.
"""

import numpy as np
import streamlit as st
from typing import Tuple, Dict, List
import io


def parse_csv_file(file_content: str, filename: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Parse a single CSV file and extract sensel data and statistics.
    
    Parameters
    ----------
    file_content : str
        The full content of the CSV file as a string
    filename : str
        Name of the file for error reporting
    
    Returns
    -------
    tuple
        (sensel_array, stats_dict) where sensel_array is the 2D pressure data
        and stats_dict contains metadata from rows 14-19
    """
    try:
        # Split into lines
        lines = file_content.strip().split('\n')
        
        # Extract rows 14–19 (1-indexed → lines[13:19]) for statistics
        stats_block = [line.strip().split(",") for line in lines[13:19]]
        
        # Convert to dict
        stats_dict = {}
        for row in stats_block:
            if len(row) >= 2:
                key = row[0].strip()
                try:
                    value = float(row[1])
                except ValueError:
                    value = row[1].strip()
                stats_dict[key] = value
        
        # Load sensel grid (after "SENSELS", starting at line 21)
        # Skip first 21 lines and parse the rest as CSV
        sensel_lines = lines[21:]
        sensel_data = []
        for line in sensel_lines:
            if line.strip():  # Skip empty lines
                row = [float(x) if x.strip() else 0 for x in line.split(',')]
                sensel_data.append(row)
        
        arr = np.array(sensel_data, dtype=float)
        
        return arr, stats_dict
        
    except Exception as e:
        raise Exception(f"Error parsing {filename}: {e}")


def load_csv_files(uploaded_files) -> Tuple[np.ndarray, Dict[str, List], str, List[str]]:
    """Load and process one or more CSV files from Streamlit file uploader.
    
    Parameters
    ----------
    uploaded_files : list or UploadedFile
        File(s) from st.file_uploader
    
    Returns
    -------
    tuple
        (merged_data, statistics, display_filename, filenames)
        - merged_data: np.ndarray - merged sensel data (elementwise max if multiple files)
        - statistics: dict - columnar statistics from all files
        - display_filename: str - name to display for the loaded data
        - filenames: list - list of all loaded filenames
    """
    # Handle single file vs multiple files
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    
    if not uploaded_files:
        raise Exception("No files uploaded.")
    
    data_list = []
    stats_list = []
    filenames = []
    
    for uploaded_file in uploaded_files:
        # Read file content as string
        file_content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        filename = uploaded_file.name
        filenames.append(filename)
        
        # Parse the file
        arr, stats_dict = parse_csv_file(file_content, filename)
        
        data_list.append(arr)
        stats_list.append(stats_dict)
    
    # Ensure all arrays are the same shape
    shapes = {d.shape for d in data_list}
    if len(shapes) > 1:
        raise ValueError(f"Shape mismatch across files: {shapes}")
    
    # Merge sensel data if multiple files
    if len(data_list) > 1:
        # Elementwise maximum across all files
        merged_data = np.maximum.reduce(data_list)
        display_filename = f"{len(filenames)}_files_merged"
    else:
        merged_data = data_list[0]
        display_filename = filenames[0] 
    
    # Rotate 180 degrees by default before storing
    merged_data = np.rot90(merged_data, k=2)
    
    # Build columnar statistics table
    all_keys = list(stats_list[0].keys())
    statistics = {key: [stats[key] for stats in stats_list] for key in all_keys}
    
    return merged_data, statistics, display_filename, filenames


def store_csv_data_in_session(
    data: np.ndarray,
    statistics: Dict[str, List],
    filename: str,
    filenames: List[str],
    side_key: str = "sleeper_1"
) -> None:
    """Store loaded CSV data in Streamlit session state.
    
    Parameters
    ----------
    data : np.ndarray
        The merged sensel pressure data
    statistics : dict
        Columnar statistics from all files
    filename : str
        Display name for the dataset
    filenames : list
        List of all source filenames
    side_key : str
        Which sleeper to associate this data with ("sleeper_1" or "sleeper_2")
    """
    if "csv_data" not in st.session_state:
        st.session_state.csv_data = {}
    
    st.session_state.csv_data[side_key] = {
        "sensel_data": data,
        "statistics": statistics,
        "filename": filename,
        "filenames": filenames,
        "shape": data.shape,
        "value_range": (data.min(), data.max())
    }


def get_csv_data_from_session(side_key: str = "sleeper_1") -> Dict:
    """Retrieve stored CSV data from session state.
    
    Parameters
    ----------
    side_key : str
        Which sleeper's data to retrieve ("sleeper_1" or "sleeper_2")
    
    Returns
    -------
    dict or None
        The stored CSV data dictionary, or None if not found
    """
    if "csv_data" not in st.session_state:
        return None
    
    return st.session_state.csv_data.get(side_key)


def clear_csv_data(side_key: str = None) -> None:
    """Clear CSV data from session state and reset curve to manual mode.
    
    This will:
    1. Remove the CSV pressure map data
    2. Clear spline mode flags and knot data
    3. Reset the master array to a flat array based on current firmness
    
    Parameters
    ----------
    side_key : str, optional
        Specific sleeper to clear. If None, clears all CSV data.
    """
    if "csv_data" not in st.session_state:
        st.session_state.csv_data = {}
    
    if side_key is None:
        # Clear all CSV data
        st.session_state.csv_data = {}
        # Reset both sleepers to manual mode
        for key in ["sleeper_1", "sleeper_2"]:
            _reset_to_manual_mode(key)
    elif side_key in st.session_state.csv_data:
        # Clear specific sleeper's CSV data
        del st.session_state.csv_data[side_key]
        # Reset this sleeper to manual mode
        _reset_to_manual_mode(side_key)


def _reset_to_manual_mode(side_key: str) -> None:
    """Reset a sleeper's curve to manual mode with flat array at current firmness.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    """
    import data_processing as dp
    
    if "answers" not in st.session_state:
        return
    
    if side_key not in st.session_state.answers:
        return
    
    sleeper_data = st.session_state.answers[side_key]
    
    # Get current firmness value (default to 2 if not set)
    firmness = sleeper_data.get("firmness_value", 2)
    
    # Clear spline mode flags and data
    sleeper_data.pop("use_scipy_spline", None)
    sleeper_data.pop("spline_knots", None)
    sleeper_data.pop("spline_smoothing", None)
    sleeper_data.pop("curve_scale_percent", None)
    sleeper_data.pop("curve_control_points", None)
    sleeper_data.pop("original_pressure_1d", None)
    sleeper_data.pop("original_firmness", None)
    
    # Reset to default 6 control points
    sleeper_data["num_control_points"] = 6
    
    # Reinitialize master array to flat array at current firmness
    array_length = dp.get_array_length()
    master_array = dp.initialize_master_array(side_key, firmness, array_length)
    dp.set_master_array(side_key, master_array)
