import streamlit as st
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components
from pathlib import Path
import common as ui
import data_processing as dp

ui.apply_base_ui("Pixel Sandbox")

# Initialize session state if not already done
if "answers" not in st.session_state:
    st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}

# Initialize master arrays from data_processing
dp.initialize_sleeper_master_arrays()

# Initialize bed size configuration
st.session_state.setdefault("bed_size", "King")

# Get current bed dimensions
bed_length, bed_width = dp.get_bed_size()

# Initialize sandbox grid - COMPLETELY SEPARATE from master arrays
# Use a unique key that won't conflict with anything else
if "sandbox_pixel_grid" not in st.session_state:
    st.session_state.sandbox_pixel_grid = np.full((bed_length, bed_width), 2, dtype=int)
elif st.session_state.sandbox_pixel_grid.shape != (bed_length, bed_width):
    # Resize if bed size changed
    st.session_state.sandbox_pixel_grid = np.full((bed_length, bed_width), 2, dtype=int)

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

# Main content area
st.markdown("---")

st.subheader(f"Interactive Pixel Grid ({st.session_state.bed_size}: {bed_length} rows × {bed_width} columns)")

# Add debug info
st.caption(f"Grid shape: {st.session_state.sandbox_pixel_grid.shape}, Unique values: {np.unique(st.session_state.sandbox_pixel_grid)}")

# Color mapping for firmness levels (0-4)
color_map = {
    0: "#E9F1F0",  # Very Soft (Teal)
    1: "#A7C7BF",  # Soft (Tibetan Stone)
    2: "#1E99A8",  # Medium (Teal/Blue)
    3: "#006261",  # Firm (Dark Teal)
    4: "#0A2734",  # Very Firm (Darkest)
}

st.caption("Click any cell to cycle through firmness levels (0→1→2→3→4→0)")

# Global CSS reset for Streamlit buttons
st.markdown(
    """
    <style>
    /* Reset Streamlit button defaults */
    .pixel-grid-wrapper div[data-testid="stButton"] button {
        all: unset !important;
        box-sizing: border-box !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 35px !important;
        height: 35px !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 1px solid #333 !important;
        font-size: 12px !important;
        font-weight: bold !important;
        font-family: monospace !important;
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .pixel-grid-wrapper div[data-testid="stButton"] button:hover {
        border: 2px solid white !important;
        transform: scale(1.08) !important;
        z-index: 10 !important;
    }
    .pixel-grid-wrapper div[data-testid="stButton"] button:active {
        transform: scale(0.96) !important;
    }
    /* Tighten column gaps */
    .pixel-grid-wrapper div[data-testid="stHorizontalBlock"] {
        gap: 2px !important;
        justify-content: flex-start !important;
        width: fit-content !important;
    }
    .pixel-grid-wrapper div[data-testid="column"] {
        padding: 0 !important;
        flex: 0 0 auto !important;
        width: 35px !important;
        min-width: 35px !important;
        max-width: 35px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="pixel-grid-wrapper">', unsafe_allow_html=True)

# Initialize selection state
if "selected_cells" not in st.session_state:
    st.session_state.selected_cells = set()

# Render grid row by row with scoped CSS per button
for row in range(bed_length):
    cols = st.columns(bed_width, gap="small")
    for col in range(bed_width):
        with cols[col]:
            val = int(st.session_state.sandbox_pixel_grid[row, col])
            bg_color = color_map[val]
            # Scoped wrapper for this cell
            st.markdown(f'<div id="cell_{row}_{col}" class="cell-wrap" style="display:inline-block;">', unsafe_allow_html=True)
            # Determine if selected
            is_selected = (row, col) in st.session_state.selected_cells
            border_color = "#FFD54F" if is_selected else "#333"
            border_width = "2px" if is_selected else "1px"
            # Scoped CSS targeting only the button inside this wrapper
            st.markdown(
                f"""
                <style>
                #cell_{row}_{col} div[data-testid="stButton"] button {{
                    background: {bg_color} !important;
                    background-color: {bg_color} !important;
                    background-image: none !important;
                    box-shadow: none !important;
                    outline: none !important;
                    border: {border_width} solid {border_color} !important;
                }}
                #cell_{row}_{col} div[data-testid="stButton"] button:hover {{
                    background: {bg_color} !important;
                    background-color: {bg_color} !important;
                    background-image: none !important;
                    box-shadow: none !important;
                    outline: none !important;
                    border: {border_width} solid {border_color} !important;
                }}
                #cell_{row}_{col} div[data-testid="stButton"] button:active {{
                    background: {bg_color} !important;
                    background-color: {bg_color} !important;
                    background-image: none !important;
                    box-shadow: none !important;
                    outline: none !important;
                    border: {border_width} solid {border_color} !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            # The actual button toggles selection (no value change)
            if st.button(str(val), key=f"cell_{row}_{col}"):
                cell = (row, col)
                if cell in st.session_state.selected_cells:
                    st.session_state.selected_cells.remove(cell)
                else:
                    st.session_state.selected_cells.add(cell)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Controls to apply value to selected cells
st.markdown("---")
apply_col1, apply_col2, apply_col3 = st.columns([1,1,2])
with apply_col1:
    new_val = st.selectbox("Set value", options=[0,1,2,3,4], index=2)
with apply_col2:
    if st.button("Apply to selected", key="apply_selected"):
        for (r, c) in list(st.session_state.selected_cells):
            st.session_state.sandbox_pixel_grid[r, c] = new_val
        # Clear selection after apply
        st.session_state.selected_cells.clear()
        st.rerun()

# Bottom navigation buttons
st.markdown("---")

col_prev, col_spacer, col_export = st.columns([1, 2, 1])

with col_prev:
    if st.button("← Previous", width="content", type="secondary"):
        st.switch_page("pages/3_Configure.py")

with col_export:
    if st.button("Export PDF", width="content", type="primary"):
        try:
            import exporter
            
            # Show loading spinner while generating PDF
            with st.spinner("Generating PDF report..."):
                # Generate PDF report
                pdf_bytes = exporter.generate_pdf_report()
                
                if pdf_bytes:
                    # Create download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"lovesac_mattress_config_{timestamp}.pdf"
                    
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        width="content",
                        type="primary"
                    )
                    st.success("PDF generated successfully!")
                else:
                    st.error("Failed to generate PDF. Please try again.")
        except ImportError:
            st.warning("PDF export functionality not yet implemented.")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
