# Dual-Mode Curve Editor System

## Overview

The curve editor now supports two distinct modes of operation:

1. **Manual Mode** (default): Uses fixed control points with Hermite interpolation
2. **Spline Mode** (pressure data): Uses scipy spline with variable knots

## Mode Determination

The system automatically switches modes based on whether pressure map data has been uploaded:

- **Manual Mode**: No pressure map uploaded, or `use_scipy_spline` flag is False
- **Spline Mode**: Pressure map uploaded and fitted with scipy spline

Check mode status with: `_is_spline_mode(side_key)`

## Manual Mode

### Characteristics
- **Control Points**: Fixed 6 points (default), evenly spaced
- **Interpolation**: Piecewise cubic Hermite spline
- **Visual Style**: Blue curve, circle markers
- **Storage**: Values stored in `master_array` at control point positions

### User Workflow
1. User sets base firmness level
2. User adjusts individual control points with +/- buttons
3. Changes update `master_array` directly at control point positions
4. Hermite interpolation fills in values between control points

## Spline Mode

### Characteristics
- **Control Points**: Variable number of knots (determined by scipy spline fitting)
- **Interpolation**: Scipy UnivariateSpline (cubic, s=0 for exact fit)
- **Visual Style**: Red curve, diamond markers, labeled as "Knot" instead of "Point"
- **Storage**: Knot positions and values stored in `spline_knots` dict

### User Workflow
1. User uploads pressure map CSV
2. System fits scipy spline to pressure data (smoothing_factor = array_length * 0.5)
3. System extracts internal knots from the spline
4. Knots become the editable control points
5. User can adjust knot values with +/- buttons
6. Changes rebuild the spline with s=0 (exact interpolation through edited knots)
7. Master array is regenerated from the updated spline

### Pressure Data Flow
```
Pressure Map (2D)
  ↓ downsample to target rows
Downsampled Map (17x9)
  ↓ max per row
1D Array (17 values)
  ↓ normalize to remap range (1-3 default)
Normalized Array
  ↓ INVERT (high pressure → soft/low, low pressure → firm/high)
Inverted Array
  ↓ fit scipy spline (smoothing_factor = array_length * 0.5)
Scipy Spline
  ↓ extract internal knots
Knot Positions & Values
  ↓ store in session state
spline_knots: {x: [positions], y: [values]}
  ↓ evaluate spline
Master Array (integer values 0-4)
```

## Session State Structure

### Manual Mode
```python
st.session_state.answers[side_key] = {
    "master_array": [int, int, ...],  # Length = array_length
    "num_control_points": 6,          # Default fixed count
    "firmness_value": 2,              # Base firmness
    "remap_range": "low",             # "extra_low", "low", or "high"
    # Optional legacy support:
    "curve_control_points": {"x": [...], "y": [...]}
}
```

### Spline Mode
```python
st.session_state.answers[side_key] = {
    "master_array": [int, int, ...],  # Generated from spline
    "use_scipy_spline": True,         # MODE FLAG
    "spline_knots": {
        "x": [int, int, ...],         # 1-indexed positions
        "y": [float, float, ...]      # Values at knots
    },
    "spline_smoothing": float,        # Smoothing factor used
    "num_control_points": int,        # Number of knots (variable)
    "remap_range": "low",             # "extra_low", "low", or "high"
    "firmness_value": 2,              # Base firmness (preserved)
}
```

## Key Functions

### data_processing.py

#### `apply_pressure_map_to_curve()`
- Converts 2D pressure map to 1D array
- Normalizes and inverts to firmness values
- Fits scipy spline to data
- **Extracts internal knots** using `spline.get_knots()`
- Stores knot positions (1-indexed) and values
- Sets `use_scipy_spline = True` flag
- Generates initial `master_array` from spline

### curve_editor.py

#### `_is_spline_mode(side_key)`
- Returns `True` if `use_scipy_spline` flag is set
- Used throughout to determine behavior

#### `_get_curve_data(side_key, ...)`
- **Spline Mode**: Returns knot positions and values from `spline_knots`
- **Manual Mode**: Returns evenly-spaced control points sampled from `master_array`

#### `get_interpolated_curve_lut(side_key, ...)`
- **Spline Mode**: Rebuilds scipy spline from knots (s=0), evaluates at all positions
- **Manual Mode**: Uses Hermite interpolation with control points

#### `show_curve_controls(side_key, ...)`
- Renders +/- buttons for each control point/knot
- **Spline Mode**: Updates `spline_knots["y"]`, rebuilds spline, regenerates `master_array`
- **Manual Mode**: Updates `master_array` directly at control point positions

#### `show_curve_plot(side_key, ...)`
- **Spline Mode**: Uses scipy spline for curve display, red diamonds for knots
- **Manual Mode**: Uses Hermite interpolation, blue circles for control points
- Pressure data overlay (red dots) only shown in manual mode to avoid confusion

## Visual Distinctions

| Feature | Manual Mode | Spline Mode |
|---------|-------------|-------------|
| Curve Color | Blue (#0492a8) | Red (#d62728) |
| Marker Shape | Circle | Diamond |
| Marker Size | 8 | 10 |
| Label | "Point" | "Knot" |
| Control Count | Fixed (6) | Variable (from scipy) |
| Pressure Overlay | Shown (red dots) | Hidden (curve IS the data) |

## Remap Ranges

Three remap ranges control the firmness value spread:

- **Extra Low** (1-2): Narrow range, prevents very soft or very firm
- **Low** (1-3): Default, moderate range
- **High** (0-4): Full range, allows extremes

All ranges are inverted after normalization:
- High pressure in data → Soft (low firmness value)
- Low pressure in data → Firm (high firmness value)

## Testing Scenarios

### Scenario 1: Fresh Start (Manual Mode)
1. User answers questions, selects firmness
2. System creates evenly-spaced 6 control points
3. User adjusts control points
4. Blue curve with circles

### Scenario 2: Upload Pressure Data (Switch to Spline Mode)
1. User uploads pressure map CSV
2. System fits scipy spline, extracts ~8-12 knots (varies)
3. Curve switches to red with diamonds
4. User can still adjust knot values
5. Adjustments rebuild spline exactly through new knot positions

### Scenario 3: Adjust After Upload (Spline Mode Editing)
1. User in spline mode
2. User clicks +/- on a knot
3. System updates that knot's Y value
4. System rebuilds spline with s=0 (exact fit through all knots)
5. System regenerates entire `master_array` from new spline
6. Curve updates smoothly

## Implementation Notes

- Knots are stored as 1-indexed positions to match "Row" display
- Scipy operations use 0-indexed positions (subtract 1 before fitting)
- Smoothing factor for initial fit: `array_length * 0.5` (moderate)
- Exact interpolation for edited knots: `s=0` (no smoothing)
- Spline degree is adaptive: `k = min(3, len(knots) - 1)` to handle small knot counts
- Master array is always integer (0-4), but knots store float values for precision

## Benefits

1. **Accuracy**: Spline mode captures pressure data nuances with variable knot positions
2. **Editability**: Users can still adjust knots after fitting
3. **Simplicity**: Manual mode keeps the simple 6-point interface
4. **Seamless**: Mode switching is automatic based on data availability
5. **Visual Clarity**: Red/diamond vs blue/circle makes mode instantly recognizable
