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
    
    # Extract key information
    name = sleeper_data.get("setting1", sleeper_key.replace("_", " ").title())
    height = sleeper_data.get("setting2", "Unknown")
    firmness_value = sleeper_data.get("firmness_value", 2)
    notes = sleeper_data.get("notes", "")
    curve_scale = sleeper_data.get("curve_scale_percent", 50)
    smoothing = sleeper_data.get("spline_smoothing", 0.0)
    
    # Get bed size info
    bed_size = st.session_state.get("bed_size", "queen")
    array_length = dp.get_array_length()
    
    # Layout parameters
    margin = 0.5 * inch
    y_cursor = page_height - margin
    
    # === HEADER SECTION ===
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(margin, y_cursor, f"{name}")
    y_cursor -= 0.3 * inch
    
    pdf.setFont("Helvetica", 10)
    pdf.setFillColorRGB(0.4, 0.4, 0.4)
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    pdf.drawString(margin, y_cursor, f"Generated on {timestamp}")
    y_cursor -= 0.5 * inch
    
    # === SLEEPER INFO BOX ===
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(margin, y_cursor, "Sleeper Information")
    y_cursor -= 0.2 * inch
    
    pdf.setFont("Helvetica", 10)
    info_lines = [
        f"Height: {height} inches",
        f"Base Firmness: {firmness_value} ({_firmness_label(firmness_value)})",
        f"Bed Size: {bed_size.title()}",
        f"Array Length: {array_length} rows",
        f"Firmness Range: {curve_scale}%",
        f"Curve Smoothing: {smoothing:.2f}%",
    ]
    
    for line in info_lines:
        pdf.drawString(margin + 0.2 * inch, y_cursor, line)
        y_cursor -= 0.18 * inch
    
    if notes and notes.strip():
        y_cursor -= 0.1 * inch
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(margin, y_cursor, "Notes:")
        y_cursor -= 0.18 * inch
        pdf.setFont("Helvetica", 9)
        # Wrap notes text if too long
        wrapped_notes = _wrap_text(notes, 90)
        for note_line in wrapped_notes[:5]:  # Limit to 5 lines
            pdf.drawString(margin + 0.2 * inch, y_cursor, note_line)
            y_cursor -= 0.15 * inch
    
    y_cursor -= 0.3 * inch
    
    # === HEATMAP SECTION ===
    try:
        # Generate heatmap figure
        master_array = dp.get_master_array(sleeper_key, array_length=array_length)
        pixel_map_2d = dp.master_array_to_pixel_map(master_array)
        
        # Get custom colorscale if available
        colorscale = dp.get_firmness_colorscale()
        
        heatmap_fig = dp.draw_pixel_map(
            pixel_map_2d=pixel_map_2d,
            colorscale=colorscale,
            show_values=True,
            value_range=(0, 4),
            title=f"{name}'s Firmness Configuration",
            return_fig=True  # Request figure instead of displaying
        )
        
        # Convert Plotly figure to PNG image
        img_bytes = heatmap_fig.to_image(format="png", width=800, height=600)
        img = ImageReader(io.BytesIO(img_bytes))
        
        # Draw heatmap (scaled to fit page width)
        img_width = page_width - 2 * margin
        img_height = img_width * 0.6  # Maintain aspect ratio
        
        if y_cursor - img_height < margin:
            # Not enough space, move to section below
            y_cursor = page_height - margin - 3.5 * inch
        
        pdf.drawImage(img, margin, y_cursor - img_height, width=img_width, height=img_height, preserveAspectRatio=True)
        y_cursor -= img_height + 0.3 * inch
        
    except Exception as e:
        # Handle error gracefully
        pdf.setFont("Helvetica", 10)
        pdf.setFillColorRGB(0.8, 0, 0)
        pdf.drawString(margin, y_cursor, f"Error generating heatmap: {str(e)}")
        y_cursor -= 0.3 * inch
    
    # === CURVE PLOT SECTION ===
    try:
        # Generate curve plot figure
        curve_fig = ce.show_curve_plot(
            side_key=sleeper_key,
            array_length=array_length,
            value_range=(0, 4),
            width=800,
            height=400,
            return_fig=True  # Request figure instead of displaying
        )
        
        # Convert to PNG
        curve_bytes = curve_fig.to_image(format="png", width=800, height=400)
        curve_img = ImageReader(io.BytesIO(curve_bytes))
        
        # Draw curve plot
        curve_width = page_width - 2 * margin
        curve_height = curve_width * 0.4
        
        if y_cursor - curve_height < margin:
            # Start new page if needed
            pdf.showPage()
            y_cursor = page_height - margin
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(margin, y_cursor, f"{name} (continued)")
            y_cursor -= 0.4 * inch
        
        pdf.drawImage(curve_img, margin, y_cursor - curve_height, width=curve_width, height=curve_height, preserveAspectRatio=True)
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
