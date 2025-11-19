"""PDF Export Module for Lovesac Pixel UX Configuration

Generates multi-page PDF reports containing sleeper configurations,
heatmaps, curve plots, and survey data.
"""

import io
from datetime import datetime
from typing import List, Optional
import numpy as np
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import data_processing as dp
import curve_editor as ce


def generate_pdf_report(sleeper_keys: Optional[List[str]] = None) -> bytes:
    """Generate a PDF report with one page per sleeper.
    
    Parameters
    ----------
    sleeper_keys : List[str], optional
        List of sleeper identifiers to include (e.g., ["sleeper_1", "sleeper_2"]).
        If None, automatically includes all configured sleepers.
    
    Returns
    -------
    bytes
        PDF file content as bytes
    
    Raises
    ------
    ValueError
        If no sleepers are configured or if sleeper data is missing
    """
    # Determine which sleepers to export
    if sleeper_keys is None:
        sleeper_keys = []
        if st.session_state.answers.get("sleeper_1"):
            sleeper_keys.append("sleeper_1")
        if st.session_state.get("show_right") and st.session_state.answers.get("sleeper_2"):
            sleeper_keys.append("sleeper_2")
    
    if not sleeper_keys:
        raise ValueError("No sleepers configured for export")
    
    # Create PDF in memory
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter
    
    # Generate one page per sleeper
    for sleeper_key in sleeper_keys:
        _generate_sleeper_page(pdf, sleeper_key, page_width, page_height)
        pdf.showPage()  # Move to next page
    
    # Finalize PDF
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def _generate_sleeper_page(pdf: canvas.Canvas, sleeper_key: str, page_width: float, page_height: float):
    """Generate a single page for one sleeper.
    
    Parameters
    ----------
    pdf : canvas.Canvas
        ReportLab canvas to draw on
    sleeper_key : str
        Sleeper identifier ("sleeper_1" or "sleeper_2")
    page_width : float
        Page width in points
    page_height : float
        Page height in points
    """
    # Get sleeper data
    sleeper_data = st.session_state.answers.get(sleeper_key, {})
    if not sleeper_data:
        raise ValueError(f"No data found for {sleeper_key}")
    
    # Extract survey information
    name = sleeper_data.get("setting1", sleeper_key.replace("_", " ").title())
    height = sleeper_data.get("setting2", "Unknown")
    weight = sleeper_data.get("setting3", "Unknown")
    sleep_positions = sleeper_data.get("sleep_positions", [])
    preferred_firmness = sleeper_data.get("firmness_value", 2)
    gender = sleeper_data.get("gender", "Unknown")
    notes = sleeper_data.get("notes", "")
    
    # Extract configuration information
    current_firmness = sleeper_data.get("firmness_value", 2)
    curve_scale = sleeper_data.get("curve_scale_percent", 50)
    smoothing = sleeper_data.get("spline_smoothing", 0.0)
    bed_size = st.session_state.get("bed_size", "queen")
    array_length = dp.get_array_length()
    
    # Check if pressure data exists
    has_pressure_data = ("csv_data" in st.session_state and 
                         sleeper_key in st.session_state.csv_data and
                         "sensel_data" in st.session_state.csv_data[sleeper_key])
    
    # Layout parameters
    margin = 0.5 * inch
    sidebar_width = (page_width - 2 * margin) / 3  # Left 1/3 for information
    map_width = (page_width - 2 * margin - sidebar_width) / 2  # Middle and right thirds for maps
    
    y_cursor = page_height - margin
    
    # === HEADER SECTION ===
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(margin, y_cursor, f"{name}")
    y_cursor -= 0.25 * inch
    
    pdf.setFont("Helvetica", 9)
    pdf.setFillColorRGB(0.4, 0.4, 0.4)
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    pdf.drawString(margin, y_cursor, f"Generated on {timestamp}")
    y_cursor -= 0.4 * inch
    
    # Store the starting y position for heatmaps
    heatmap_y_start = y_cursor
    
    # === LEFT SIDEBAR: ALL INFORMATION SECTIONS ===
    sidebar_x = margin
    sidebar_y = y_cursor
    
    # SECTION 1: SLEEPER PROFILE
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(sidebar_x, sidebar_y, "Sleeper Profile")
    sidebar_y -= 0.18 * inch
    
    pdf.setFont("Helvetica", 8)
    profile_lines = [
        f"Height: {height}\"",
        f"Weight: {weight} lbs",
        f"Gender: {gender}",
        f"Sleep: {', '.join(sleep_positions) if sleep_positions else 'N/A'}",
        f"Pref. Firmness: {preferred_firmness} ({_firmness_label(preferred_firmness)})",
    ]
    
    for line in profile_lines:
        pdf.drawString(sidebar_x + 0.1 * inch, sidebar_y, line)
        sidebar_y -= 0.14 * inch
    
    if notes and notes.strip():
        sidebar_y -= 0.05 * inch
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(sidebar_x + 0.1 * inch, sidebar_y, "Notes:")
        sidebar_y -= 0.13 * inch
        pdf.setFont("Helvetica", 7)
        # Wrap notes text to fit sidebar
        wrapped_notes = _wrap_text(notes, 35)
        for note_line in wrapped_notes[:3]:  # Limit to 3 lines
            pdf.drawString(sidebar_x + 0.15 * inch, sidebar_y, note_line)
            sidebar_y -= 0.12 * inch
    
    sidebar_y -= 0.15 * inch
    
    # SECTION 2: PRESSURE MAP STATISTICS
    if has_pressure_data:
        pdf.setFont("Helvetica-Bold", 10)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.drawString(sidebar_x, sidebar_y, "Pressure Statistics")
        sidebar_y -= 0.18 * inch
        
        try:
            statistics = st.session_state.csv_data[sleeper_key].get("statistics", {})
            pdf.setFont("Helvetica", 7)
            for key, values in statistics.items():
                if len(values) == 1:
                    stat_text = f"{key}: {values[0]}"
                else:
                    # Truncate if too long
                    val_str = ', '.join(map(str, values))
                    if len(val_str) > 30:
                        val_str = val_str[:27] + "..."
                    stat_text = f"{key}: {val_str}"
                pdf.drawString(sidebar_x + 0.1 * inch, sidebar_y, stat_text)
                sidebar_y -= 0.12 * inch
        except Exception:
            pdf.setFont("Helvetica", 7)
            pdf.setFillColorRGB(0.6, 0, 0)
            pdf.drawString(sidebar_x + 0.1 * inch, sidebar_y, "Error loading stats")
            sidebar_y -= 0.12 * inch
        
        sidebar_y -= 0.1 * inch
    
    # SECTION 3: CONFIGURATION DETAILS
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawString(sidebar_x, sidebar_y, "Configuration")
    sidebar_y -= 0.18 * inch
    
    pdf.setFont("Helvetica", 8)
    config_lines = [
        f"Bed: {bed_size.title()}",
        f"Firmness: {current_firmness} ({_firmness_label(current_firmness)})",
        f"Range: {curve_scale}%",
        f"Smoothing: {smoothing:.2f}%",
    ]
    
    for line in config_lines:
        pdf.drawString(sidebar_x + 0.1 * inch, sidebar_y, line)
        sidebar_y -= 0.14 * inch
    
    # === MIDDLE: PRESSURE MAP ===
    pressure_x = margin + sidebar_width + 0.1 * inch
    map_height = map_width * 1.8  # Taller aspect ratio for vertical maps
    
    if has_pressure_data:
        try:
            # Get pressure data and downsample
            sensel_data = st.session_state.csv_data[sleeper_key]["sensel_data"]
            downsampled = dp.downsample_pressure_map(sensel_data, target_shape=(array_length, 9))
            
            # Create pressure map heatmap
            pressure_fig = dp.draw_pixel_map(
                pixel_map_2d=downsampled,
                colorscale=None,  # Use default blue gradient
                show_values=False,
                height=500,
                value_range="auto",
                title="Pressure Map",
                title_font_size=15,
                return_fig=True
            )
            
            # Convert to PNG
            pressure_bytes = pressure_fig.to_image(format="png", width=400, height=700)
            pressure_img = ImageReader(io.BytesIO(pressure_bytes))
            
            # Draw pressure map
            pdf.drawImage(pressure_img, pressure_x, heatmap_y_start - map_height, 
                         width=map_width, height=map_height, preserveAspectRatio=True)
            
        except Exception as e:
            # Show error message
            pdf.setFont("Helvetica", 8)
            pdf.setFillColorRGB(0.6, 0, 0)
            pdf.drawString(pressure_x, heatmap_y_start - 0.5 * inch, f"Error: {str(e)[:40]}")
    else:
        # Show placeholder
        pdf.setFont("Helvetica-Bold", 10)
        pdf.setFillColorRGB(0.5, 0.5, 0.5)
        pdf.drawString(pressure_x, heatmap_y_start - 0.3 * inch, "Pressure Map")
        pdf.setFont("Helvetica", 9)
        pdf.drawString(pressure_x, heatmap_y_start - 0.5 * inch, "No data uploaded")
    
    # === RIGHT: FIRMNESS CONFIGURATION ===
    firmness_x = pressure_x + map_width + 0.1 * inch
    
    try:
        # Generate the firmness array from the curve editor (same as Configure page)
        # This ensures we get the current curve state, not just the master_array
        from curve_editor import get_interpolated_curve_lut
        firmness_array = get_interpolated_curve_lut(sleeper_key, array_length=array_length)
        
        if firmness_array is None:
            # Fallback to master array if curve generation fails
            firmness_array = dp.get_master_array(sleeper_key, array_length=array_length)
        
        # Get bed width and create pixel map from the generated array
        total_width = dp.get_array_width()
        width_per_sleeper = total_width // 2
        pixel_map_2d = dp.pixel_map(firmness_array, width_per_sleeper)
        
        # Custom colorscale matching the Configure page
        colorscale = [
            [0.0, "#E9F1F0"],
            [0.25, "#A7C7BF"],
            [0.5, "#1E99A8"],
            [0.75, "#006261"],
            [1.0, "#0A2734"]
        ]
        
        heatmap_fig = dp.draw_pixel_map(
            pixel_map_2d=pixel_map_2d,
            colorscale=colorscale,
            show_values=True,
            height=500,
            value_range=(0, 4),
            title="Firmness Config",
            title_font_size=15,
            return_fig=True
        )
        
        # Convert Plotly figure to PNG
        img_bytes = heatmap_fig.to_image(format="png", width=400, height=700)
        img = ImageReader(io.BytesIO(img_bytes))
        
        # Draw firmness heatmap on right
        pdf.drawImage(img, firmness_x, heatmap_y_start - map_height, 
                     width=map_width, height=map_height, preserveAspectRatio=True)
        
    except Exception as e:
        # Handle error gracefully
        pdf.setFont("Helvetica", 8)
        pdf.setFillColorRGB(0.8, 0, 0)
        pdf.drawString(firmness_x, heatmap_y_start - 0.5 * inch, f"Error: {str(e)[:40]}")
    
    # Move cursor down past heatmaps
    y_cursor = heatmap_y_start - map_height - 0.3 * inch
    
    # === CURVE PLOT SECTION (Full Width) ===
    try:
        # Generate curve plot figure
        curve_fig = ce.show_curve_plot(
            side_key=sleeper_key,
            array_length=array_length,
            value_range=(0, 4),
            width=1000,
            height=150,
            return_fig=True
        )
        
        # Convert to PNG
        curve_bytes = curve_fig.to_image(format="png", width=1000, height=150)
        curve_img = ImageReader(io.BytesIO(curve_bytes))
        
        # Draw curve plot spanning full width
        curve_width = page_width - 2 * margin
        curve_height = curve_width * 0.15  # More compact for better fit
        
        pdf.drawImage(curve_img, margin, y_cursor - curve_height, 
                     width=curve_width, height=curve_height, preserveAspectRatio=True)
        y_cursor -= curve_height + 0.2 * inch
        
    except Exception as e:
        # Handle error gracefully
        pdf.setFont("Helvetica", 10)
        pdf.setFillColorRGB(0.8, 0, 0)
        pdf.drawString(margin, y_cursor, f"Error generating curve plot: {str(e)}")
        y_cursor -= 0.3 * inch
    
    # === FOOTER ===
    pdf.setFont("Helvetica", 8)
    pdf.setFillColorRGB(0.5, 0.5, 0.5)
    pdf.drawString(margin, 0.5 * inch, "Lovesac Pixel UX Configuration Report")
    pdf.drawRightString(page_width - margin, 0.5 * inch, f"Page for {name}")


def _firmness_label(value: int) -> str:
    """Convert firmness value to descriptive label."""
    labels = {
        0: "Very Soft",
        1: "Soft",
        2: "Medium",
        3: "Firm",
        4: "Very Firm"
    }
    return labels.get(value, "Unknown")


def _wrap_text(text: str, max_chars: int) -> List[str]:
    """Wrap text to fit within character limit per line."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > max_chars and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += word_length
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines
