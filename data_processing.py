"""Data processing utilities for the Xsensor survey application.

Manages the master 1D array for each sleeper that drives the curve editor
and other downstream visualizations.
"""

import streamlit as st
import numpy as np

# Master array configuration
MASTER_ARRAY_LENGTH = 17  # Domain 1-17 (index 0-16)
MASTER_VALUE_RANGE = (0, 4)  # Value range for array elements


def initialize_master_array(side_key: str, firmness_value: int = 2) -> np.ndarray:
    """Initialize a master 1D array for a sleeper.
    
    Creates a flat array with domain 1-17 (matching curve editor x-domain)
    initialized to the firmness value selected in the survey.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    firmness_value : int
        The initial firmness level (0-4) from the survey questions.
        This value is broadcast across all 17 array positions.
    
    Returns
    -------
    np.ndarray
        1D array of length 17 with all values set to firmness_value (as int)
    """
    # Clamp firmness value to valid range
    firmness_value = int(np.clip(firmness_value, MASTER_VALUE_RANGE[0], MASTER_VALUE_RANGE[1]))
    # Create array with domain 1-17, initialized to firmness value
    master_array = np.full(MASTER_ARRAY_LENGTH, firmness_value, dtype=int)
    return master_array


def initialize_sleeper_master_arrays():
    """Initialize master arrays in session state for all sleepers.
    
    Call this on pages where you need master array access.
    Checks if sleeper has survey data and initializes/updates the master array
    based on the firmness value from the survey.
    """
    if "answers" not in st.session_state:
        st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}
    
    # Initialize sleeper_1 master array
    if "sleeper_1" in st.session_state.answers:
        sleeper_1_data = st.session_state.answers["sleeper_1"]
        firmness_1 = sleeper_1_data.get("firmness_value", 2)  # default to 2 (middle of 0-4)
        
        if "master_array" not in sleeper_1_data:
            sleeper_1_data["master_array"] = initialize_master_array("sleeper_1", firmness_1).tolist()
        else:
            # Update master array if firmness changed
            if not isinstance(sleeper_1_data["master_array"], list):
                sleeper_1_data["master_array"] = sleeper_1_data["master_array"].tolist()
    
    # Initialize sleeper_2 master array if they've been added
    if st.session_state.get("show_right", False) and "sleeper_2" in st.session_state.answers:
        sleeper_2_data = st.session_state.answers["sleeper_2"]
        firmness_2 = sleeper_2_data.get("firmness_value", 2)  # default to 2 (middle of 0-4)
        
        if "master_array" not in sleeper_2_data:
            sleeper_2_data["master_array"] = initialize_master_array("sleeper_2", firmness_2).tolist()
        else:
            # Update master array if firmness changed
            if not isinstance(sleeper_2_data["master_array"], list):
                sleeper_2_data["master_array"] = sleeper_2_data["master_array"].tolist()


def get_master_array(side_key: str) -> np.ndarray:
    """Retrieve the master array for a sleeper as a numpy array.
    
    Parameters
    ----------
    side_key : str
        The sleeper identifier ("sleeper_1" or "sleeper_2")
    
    Returns
    -------
    np.ndarray
        The master 1D array (length 17, dtype int) as a numpy array, or default if not found
    """
    if "answers" in st.session_state and side_key in st.session_state.answers:
        sleeper_data = st.session_state.answers[side_key]
        if "master_array" in sleeper_data:
            return np.array(sleeper_data["master_array"], dtype=int)
    
    # Return default if not found
    return np.full(MASTER_ARRAY_LENGTH, MASTER_VALUE_RANGE[0], dtype=int)


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