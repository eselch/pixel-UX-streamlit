"""
Reusable Clickable Color Grid Component for Streamlit

Usage:
    from color_grid_component import render_color_grid

    # Basic usage with defaults
    render_color_grid()

    # Custom configuration
    render_color_grid(
        initial_pattern=[
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ],
        colors=["#eee", "#f00", "#0f0", "#00f"],
        color_names=["Empty", "Red", "Green", "Blue"],
        cell_size=50,
        spacing=6,
        session_key="my_grid",
    )
"""

import streamlit as st
import json


def render_color_grid(
    initial_pattern: list[list[int]] | None = None,
    colors: list[str] | None = None,
    color_names: list[str] | None = None,
    cell_size: int = 40,
    spacing: int = 4,
    session_key: str = "color_grid",
    screenshot_subtitle: str = "Sleepers 1 and 2",
    bed_size: str = "King",
    # CSS customization
    font_family: str = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    cell_border: str = "2px solid #888",
    cell_border_radius: str = "0px",
    selected_border: str = "2px solid #000",
    selected_shadow: str = "0 0 0 2px #fff, 0 0 0 4px #000",
    button_border_radius: str = "4px",
    label_color: str = "#333",
    info_color: str = "#666",
    clear_btn_bg: str = "#888",
    clear_btn_hover_bg: str = "#666",
):
    """
    Render a clickable color grid component.

    Args:
        initial_pattern: 2D list of color indices to initialize the grid.
                        If None, creates a 5x5 grid of zeros.
        colors: List of hex color strings. Default is gray, red, green, blue, yellow.
        color_names: Display names for each color. Must match length of colors.
        cell_size: Size of each cell in pixels.
        spacing: Gap between cells in pixels.
        session_key: Unique key for session state (use different keys for multiple grids).
        screenshot_subtitle: Subtitle text to display in screenshot (e.g., sleeper names).

        # CSS customization options:
        font_family: Font stack for the component.
        cell_border: Border style for grid cells.
        cell_border_radius: Border radius for grid cells.
        selected_border: Border style for selected cells.
        selected_shadow: Box shadow for selected cells.
        button_border_radius: Border radius for color buttons.
        label_color: Text color for labels.
        info_color: Text color for info text.
        clear_btn_bg: Background color for clear button.
        clear_btn_hover_bg: Hover background color for clear button.
    """

    # Defaults
    if colors is None:
        colors = [
            "#dddddd",  # 0: Gray (default/unselected)
            "#ff5555",  # 1: Red
            "#55ff55",  # 2: Green
            "#5555ff",  # 3: Blue
            "#ffff55",  # 4: Yellow
        ]

    if color_names is None:
        color_names = ["Gray", "Red", "Green", "Blue", "Yellow"]

    if initial_pattern is None:
        initial_pattern = [[0] * 5 for _ in range(5)]

    # Validate
    assert len(colors) == len(color_names), "colors and color_names must have same length"

    rows = len(initial_pattern)
    cols = len(initial_pattern[0]) if rows > 0 else 0

    # Always update session state with the provided initial_pattern
    # This ensures the grid reflects the latest data from the caller
    st.session_state[session_key] = [row[:] for row in initial_pattern]

    # Convert to JSON for JavaScript
    grid_json = json.dumps(st.session_state[session_key])
    colors_json = json.dumps(colors)
    color_names_json = json.dumps(color_names)
    screenshot_subtitle_json = json.dumps(screenshot_subtitle)
    bed_size_json = json.dumps(bed_size)

    # Build HTML/CSS/JS
    html_code = f"""
<style>
    .cg-main-container {{
        display: flex;
        gap: 60px;
        align-items: flex-start;
        justify-content: center;
        font-family: {font_family};
        width: 100%;
    }}
    .cg-grid-container {{
        display: inline-block;
        user-select: none;
    }}
    .cg-grid-row {{
        display: flex;
        margin-bottom: {spacing}px;
    }}
    .cg-grid-row:last-child {{
        margin-bottom: 0;
    }}
    .cg-grid-cell {{
        width: {cell_size}px;
        height: {cell_size}px;
        margin-right: {spacing}px;
        border: none;
        border-radius: 0px;
        cursor: pointer;
        box-sizing: border-box;
        transition: opacity 0.1s;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 400;
        font-family: sans-serif;
        color: white;
        text-shadow: 0 0 2px rgba(0,0,0,0.5);
    }}
    .cg-grid-cell:last-child {{
        margin-right: 0;
    }}
    .cg-grid-cell:hover {{
        opacity: 0.7;
    }}
    .cg-grid-cell.selected {{
        outline: 2px solid #FFD54F;
        outline-offset: -2px;
    }}
    .cg-controls {{
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-width: 150px;
    }}
    .cg-color-btn {{
        padding: 10px 20px;
        border: none;
        border-radius: 0px;
        cursor: pointer;
        font-size: 14px;
        font-weight: bold;
        font-family: sans-serif;
        background-color: #F0F2F6;
        color: #6e6f72;
        transition: background-color 0.15s, color 0.15s;
    }}
    .cg-color-btn:hover {{
        background-color: #036d7c;
        color: white;
    }}
    .cg-color-btn:active {{
        background-color: #025763;
        color: white;
    }}
    .cg-controls-label {{
        font-weight: bold;
        margin-bottom: 4px;
        color: #6e6f72;
        font-family: sans-serif;
    }}
    .cg-selection-info {{
        font-size: 12px;
        color: #6e6f72;
        margin-top: 8px;
        font-family: sans-serif;
    }}
    .cg-clear-btn {{
        padding: 10px 20px;
        background-color: #F0F2F6;
        color: #6e6f72;
        border: none;
        border-radius: 0px;
        cursor: pointer;
        font-size: 14px;
        font-weight: bold;
        font-family: sans-serif;
        margin-top: 8px;
        transition: background-color 0.15s, color 0.15s;
    }}
    .cg-clear-btn:hover {{
        background-color: #036d7c;
        color: white;
    }}
    .cg-clear-btn:active {{
        background-color: #025763;
        color: white;
    }}
    .cg-screenshot-btn {{
        background-color: #0492a8 !important;
        color: white !important;
    }}
    .cg-screenshot-btn:hover {{
        background-color: #036d7c !important;
    }}
    .cg-bundles-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #ddd;
    }}
    .cg-bundle {{
        display: inline-block;
        cursor: pointer;
        padding: 4px;
        border: 2px solid transparent;
        transition: border-color 0.15s;
    }}
    .cg-bundle.selected {{
        border-color: #FFD54F;
    }}
    .cg-bundle-row {{
        display: flex;
    }}
    .cg-bundle-cell {{
        width: {cell_size // 2}px;
        height: {cell_size // 2}px;
        border: 1px solid white;
        box-sizing: border-box;
    }}
    .cg-bundles-label {{
        font-weight: bold;
        color: #6e6f72;
        font-family: sans-serif;
        font-size: 12px;
        width: 100%;
        margin-bottom: 4px;
    }}
</style>

<div class="cg-main-container">
    <div class="cg-grid-container" id="cgGridContainer"></div>
    <div class="cg-controls" id="cgControls">
        <div class="cg-controls-label">Set Firmness:</div>
        <div id="cgColorButtons"></div>
        <div class="cg-selection-info" id="cgSelectionInfo">Click cells to select</div>
        <button class="cg-clear-btn" id="cgClearSelection">Clear Selection</button>
        <button class="cg-clear-btn" id="cgPixelAssortment" style="margin-top: 16px;">Add Pixel Assortment</button>
        <button class="cg-clear-btn" id="cgAddBundle" style="margin-top: 8px;">+ Add Bundle</button>
        <button class="cg-clear-btn" id="cgRemoveBundle" style="margin-top: 4px;">- Remove Bundle</button>
        <button class="cg-clear-btn" id="cgCaptureScreenshot" style="margin-top: 16px; background-color: #0492a8; color: white;">� Export</button>
        <div class="cg-bundles-container" id="cgBundlesContainer">
            <div class="cg-bundles-label">Bundles:</div>
        </div>
    </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
(function() {{
    const grid = {grid_json};
    const colors = {colors_json};
    const colorNames = {color_names_json};
    const numColors = colors.length;
    const rows = grid.length;
    const cols = grid[0].length;

    const container = document.getElementById('cgGridContainer');
    const colorButtonsDiv = document.getElementById('cgColorButtons');
    const selectionInfo = document.getElementById('cgSelectionInfo');
    const clearBtn = document.getElementById('cgClearSelection');
    const pixelAssortmentBtn = document.getElementById('cgPixelAssortment');
    const addBundleBtn = document.getElementById('cgAddBundle');
    const removeBundleBtn = document.getElementById('cgRemoveBundle');
    const captureBtn = document.getElementById('cgCaptureScreenshot');
    const bundlesContainer = document.getElementById('cgBundlesContainer');

    const selected = new Set();
    const cellElements = {{}};
    
    // Bundle tracking
    const bundles = [];
    let selectedBundle = null;
    const bundleSize = Math.floor(cols / 2); // Half bed width

    // Build color buttons (no background color - use Streamlit-style buttons)
    for (let i = 0; i < numColors; i++) {{
        const btn = document.createElement('button');
        btn.className = 'cg-color-btn';
        btn.textContent = colorNames[i];
        btn.addEventListener('click', function() {{
            // Apply to selected cells in main grid
            selected.forEach(key => {{
                const [r, c] = key.split(',').map(Number);
                grid[r][c] = i;
                cellElements[key].style.backgroundColor = colors[i];
                cellElements[key].textContent = i;
            }});
            // Apply to selected bundle
            if (selectedBundle !== null && bundles[selectedBundle]) {{
                const bundle = bundles[selectedBundle];
                for (let r = 0; r < bundle.data.length; r++) {{
                    for (let c = 0; c < bundle.data[r].length; c++) {{
                        bundle.data[r][c] = i;
                        bundle.cells[r][c].style.backgroundColor = colors[i];
                    }}
                }}
            }}
        }});
        colorButtonsDiv.appendChild(btn);
    }}

    function updateSelectionInfo() {{
        const count = selected.size;
        let text = count === 0 ? 'Click cells to select' : count + ' cell(s) selected';
        if (selectedBundle !== null) {{
            text += ' | Bundle ' + (selectedBundle + 1) + ' selected';
        }}
        selectionInfo.textContent = text;
    }}

    clearBtn.addEventListener('click', function() {{
        selected.forEach(key => {{
            cellElements[key].classList.remove('selected');
        }});
        selected.clear();
        // Deselect bundle
        if (selectedBundle !== null && bundles[selectedBundle]) {{
            bundles[selectedBundle].element.classList.remove('selected');
            selectedBundle = null;
        }}
        updateSelectionInfo();
    }});

    // Remove Bundle functionality
    removeBundleBtn.addEventListener('click', function() {{
        if (selectedBundle !== null && bundles[selectedBundle]) {{
            // Remove the bundle element from DOM
            bundles[selectedBundle].element.remove();
            // Mark as removed (set to null to preserve indices)
            bundles[selectedBundle] = null;
            selectedBundle = null;
            updateSelectionInfo();
        }}
    }});

    // Add Bundle functionality
    addBundleBtn.addEventListener('click', function() {{
        const bundleIndex = bundles.length;
        const bundleData = [];
        const bundleCells = [];
        
        // Create bundle element
        const bundleEl = document.createElement('div');
        bundleEl.className = 'cg-bundle';
        
        // Create bundle grid (bundleSize x bundleSize, default firmness 2)
        for (let r = 0; r < bundleSize; r++) {{
            const rowData = [];
            const rowCells = [];
            const rowEl = document.createElement('div');
            rowEl.className = 'cg-bundle-row';
            
            for (let c = 0; c < bundleSize; c++) {{
                const cellEl = document.createElement('div');
                cellEl.className = 'cg-bundle-cell';
                cellEl.style.backgroundColor = colors[2]; // Default medium
                rowEl.appendChild(cellEl);
                rowData.push(2);
                rowCells.push(cellEl);
            }}
            bundleEl.appendChild(rowEl);
            bundleData.push(rowData);
            bundleCells.push(rowCells);
        }}
        
        // Click to select/deselect bundle
        bundleEl.addEventListener('click', function() {{
            if (selectedBundle === bundleIndex) {{
                // Deselect
                bundleEl.classList.remove('selected');
                selectedBundle = null;
            }} else {{
                // Deselect previous bundle
                if (selectedBundle !== null && bundles[selectedBundle]) {{
                    bundles[selectedBundle].element.classList.remove('selected');
                }}
                // Select this bundle
                bundleEl.classList.add('selected');
                selectedBundle = bundleIndex;
            }}
            updateSelectionInfo();
        }});
        
        bundles.push({{
            element: bundleEl,
            data: bundleData,
            cells: bundleCells
        }});
        
        bundlesContainer.appendChild(bundleEl);
    }});

    // Pixel Assortment: set bottom 5 rows to firmness 0-4
    pixelAssortmentBtn.addEventListener('click', function() {{
        const startRow = Math.max(0, rows - 5);
        for (let r = startRow; r < rows; r++) {{
            const firmness = r - startRow; // 0, 1, 2, 3, 4
            for (let c = 0; c < cols; c++) {{
                const key = r + ',' + c;
                grid[r][c] = firmness;
                if (cellElements[key]) {{
                    cellElements[key].style.backgroundColor = colors[firmness];
                    cellElements[key].textContent = firmness;
                }}
            }}
        }}
    }});

    // Screenshot capture functionality
    captureBtn.addEventListener('click', function() {{
        // Temporarily remove selection highlights for cleaner screenshot
        const selectedCells = document.querySelectorAll('.cg-grid-cell.selected');
        selectedCells.forEach(cell => cell.classList.remove('selected'));
        const selectedBundleEl = document.querySelector('.cg-bundle.selected');
        if (selectedBundleEl) selectedBundleEl.classList.remove('selected');
        
        // Capture the main container (grid + controls + bundles)
        const mainContainer = document.querySelector('.cg-main-container');
        
        html2canvas(mainContainer, {{
            backgroundColor: '#ffffff',
            scale: 2, // Higher resolution
            logging: false,
            useCORS: true
        }}).then(function(canvas) {{
            // Restore selection highlights
            selectedCells.forEach(cell => cell.classList.add('selected'));
            if (selectedBundleEl) selectedBundleEl.classList.add('selected');
            
            // Create a new canvas with padding and title
            const padding = 80;
            const titleHeight = 100;
            const newCanvas = document.createElement('canvas');
            newCanvas.width = canvas.width + (padding * 2);
            newCanvas.height = canvas.height + (padding * 2) + titleHeight;
            const ctx = newCanvas.getContext('2d');
            
            // Fill white background
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, newCanvas.width, newCanvas.height);
            
            // Format date and time
            const now = new Date();
            const dateStr = now.toLocaleDateString('en-US', {{ year: 'numeric', month: 'long', day: 'numeric' }});
            const timeStr = now.toLocaleTimeString('en-US', {{ hour: '2-digit', minute: '2-digit' }});
            
            // Draw title with bed size
            ctx.fillStyle = '#333333';
            ctx.font = 'bold 32px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText('Pixel Sandbox - ' + {bed_size_json}, padding, padding + 30);
            
            // Draw subtitle (sleeper names)
            ctx.fillStyle = '#666666';
            ctx.font = '20px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText({screenshot_subtitle_json}, padding, padding + 60);
            
            // Draw date and time
            ctx.fillStyle = '#999999';
            ctx.font = '16px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif';
            ctx.fillText(dateStr + ' at ' + timeStr, padding, padding + 85);
            
            // Draw the captured content
            ctx.drawImage(canvas, padding, padding + titleHeight);
            
            // Convert to base64
            const dataURL = newCanvas.toDataURL('image/png');
            
            // Trigger download
            const link = document.createElement('a');
            link.download = 'pixel_sandbox_' + new Date().toISOString().replace(/[:.]/g, '-') + '.png';
            link.href = dataURL;
            link.click();
            
            // Visual feedback
            captureBtn.textContent = '✓ Exported!';
            captureBtn.style.backgroundColor = '#28a745';
            setTimeout(function() {{
                captureBtn.textContent = '📤 Export';
                captureBtn.style.backgroundColor = '#0492a8';
            }}, 2000);
        }}).catch(function(err) {{
            console.error('Export failed:', err);
            captureBtn.textContent = '❌ Failed';
            setTimeout(function() {{
                captureBtn.textContent = '📤 Export';
            }}, 2000);
        }});
    }});

    // Track drag state and mode (selecting or deselecting)
    let isDragging = false;
    let dragMode = null; // 'select' or 'deselect'

    // Build grid
    for (let r = 0; r < rows; r++) {{
        const rowDiv = document.createElement('div');
        rowDiv.className = 'cg-grid-row';

        for (let c = 0; c < cols; c++) {{
            const cell = document.createElement('div');
            cell.className = 'cg-grid-cell';
            cell.dataset.r = r;
            cell.dataset.c = c;
            cell.style.backgroundColor = colors[grid[r][c]];
            cell.textContent = grid[r][c];

            const key = r + ',' + c;
            cellElements[key] = cell;

            function toggleCell(key, cell) {{
                if (selected.has(key)) {{
                    selected.delete(key);
                    cell.classList.remove('selected');
                }} else {{
                    selected.add(key);
                    cell.classList.add('selected');
                }}
                updateSelectionInfo();
            }}

            // Start drag on mousedown - determine mode based on first cell's state
            cell.addEventListener('mousedown', function(e) {{
                e.preventDefault();
                isDragging = true;
                // If cell is selected, we're deselecting; otherwise selecting
                dragMode = selected.has(key) ? 'deselect' : 'select';
                toggleCell(key, cell);
            }});

            // Toggle while dragging based on drag mode
            cell.addEventListener('mouseenter', function(e) {{
                if (isDragging) {{
                    if (dragMode === 'select' && !selected.has(key)) {{
                        selected.add(key);
                        cell.classList.add('selected');
                        updateSelectionInfo();
                    }} else if (dragMode === 'deselect' && selected.has(key)) {{
                        selected.delete(key);
                        cell.classList.remove('selected');
                        updateSelectionInfo();
                    }}
                }}
            }});

            rowDiv.appendChild(cell);
        }}
        container.appendChild(rowDiv);
    }}

    // Stop drag on mouseup anywhere
    document.addEventListener('mouseup', function() {{
        isDragging = false;
        dragMode = null;
    }});

    // Set frame height
    const height = rows * ({cell_size} + {spacing}) + 20;
    window.parent.postMessage({{
        type: 'streamlit:setFrameHeight',
        height: height
    }}, '*');
}})();
</script>
"""

    # Calculate and render
    component_height = rows * (cell_size + spacing) + 50
    st.components.v1.html(html_code, height=component_height, scrolling=False)
