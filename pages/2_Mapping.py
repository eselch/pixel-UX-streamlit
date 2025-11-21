import streamlit as st
import numpy as np
import common as ui
import csv_loader
import data_processing as dp

ui.apply_base_ui("Mapping")

# Initialize session state if not already done
if "answers" not in st.session_state:
    st.session_state.answers = {"sleeper_1": {}, "sleeper_2": {}}

st.write("Upload pressure map CSV files for your profiles. You can upload multiple files to merge them.")

# Get names for labels
left_name = st.session_state.answers.get("sleeper_1", {}).get("setting1", "Sleeper 1")
right_name = st.session_state.answers.get("sleeper_2", {}).get("setting1", "Sleeper 2")

# Render dual column headers
col_left, col_right = ui.render_dual_column_headers(left_name, right_name)

with col_left:
    
    # Check if data is already loaded
    existing_data = csv_loader.get_csv_data_from_session("sleeper_1")
    
    if existing_data:
        # Downsample the pressure map
        downsampled = dp.downsample_pressure_map(existing_data['sensel_data'], target_shape=(17, 9))

        # Display the downsampled pressure map heatmap
        dp.draw_pixel_map(downsampled, width=300, show_values=True, value_range="auto", colorscale="OrRd")
        
        # Show uploaded file info and delete button
        st.write(f"**Files:** {', '.join(existing_data['filenames'])}")
        
        if st.button("Delete and Re-upload", key="delete_left", use_container_width=True):
            csv_loader.clear_csv_data("sleeper_1")
            st.rerun()
        
        # Flip button to rotate 180 degrees
        if st.button("Flip 180°", key="flip_left", use_container_width=True):
            # Rotate the original data 180 degrees
            flipped_data = np.rot90(existing_data['sensel_data'], k=2)
            # Update the stored data
            csv_loader.store_csv_data_in_session(
                flipped_data,
                existing_data['statistics'],
                existing_data['filename'],
                existing_data['filenames'],
                side_key="sleeper_1"
            )
            
            # Reapply to curve editor with flipped data
            downsampled = dp.downsample_pressure_map(flipped_data, target_shape=(17, 9))
            dp.apply_pressure_map_to_curve(downsampled, side_key="sleeper_1", num_control_points=6)
            
            st.rerun()
        
        # Display statistics in expander
        with st.expander("View Pressure Map Statistics"):
            for key, values in existing_data['statistics'].items():
                if len(values) == 1:
                    st.write(f"- {key}: {values[0]}")
                else:
                    st.write(f"- {key}: {values}")
    else:
        # Show file uploader
        left_files = st.file_uploader(
            "Choose CSV file(s) for Sleeper 1",
            type="csv",
            key="left_csv",
            label_visibility="collapsed",
            accept_multiple_files=True
        )
        
        if left_files:
            try:
                # Load and process the CSV files
                merged_data, statistics, filename, filenames = csv_loader.load_csv_files(left_files)
                
                # Store in session state
                csv_loader.store_csv_data_in_session(
                    merged_data, 
                    statistics, 
                    filename, 
                    filenames, 
                    side_key="sleeper_1"
                )
                
                # Downsample and apply to curve editor
                downsampled = dp.downsample_pressure_map(merged_data, target_shape=(17, 9))
                dp.apply_pressure_map_to_curve(downsampled, side_key="sleeper_1", num_control_points=6)
                
                st.rerun()
                            
            except Exception as e:
                st.error(f"Error loading files: {e}")

with col_right:
    # Only show right side if show_right is enabled
    if st.session_state.get("show_right", False):
        
        # Check if data is already loaded
        existing_data = csv_loader.get_csv_data_from_session("sleeper_2")
        
        if existing_data:
            # Downsample the pressure map
            downsampled = dp.downsample_pressure_map(existing_data['sensel_data'], target_shape=(17, 9))

            # Display the downsampled pressure map heatmap
            dp.draw_pixel_map(downsampled, width=300, show_values=True, value_range="auto", colorscale="OrRd")         # Show uploaded file info and delete button
            st.write(f"**Files:** {', '.join(existing_data['filenames'])}")
            
            if st.button("Delete and Re-upload", key="delete_right", use_container_width=True):
                csv_loader.clear_csv_data("sleeper_2")
                st.rerun()
            
            # Flip button to rotate 180 degrees
            if st.button("Flip 180°", key="flip_right", use_container_width=True):
                # Rotate the original data 180 degrees
                flipped_data = np.rot90(existing_data['sensel_data'], k=2)
                # Update the stored data
                csv_loader.store_csv_data_in_session(
                    flipped_data,
                    existing_data['statistics'],
                    existing_data['filename'],
                    existing_data['filenames'],
                    side_key="sleeper_2"
                )
                
                # Reapply to curve editor with flipped data
                downsampled = dp.downsample_pressure_map(flipped_data, target_shape=(17, 9))
                dp.apply_pressure_map_to_curve(downsampled, side_key="sleeper_2", num_control_points=6)
                
                st.rerun()
            
            # Display statistics in expander
            with st.expander("View Pressure Map Statistics"):
                for key, values in existing_data['statistics'].items():
                    if len(values) == 1:
                        st.write(f"- {key}: {values[0]}")
                    else:
                        st.write(f"- {key}: {values}")
        else:
            # Show file uploader
            right_files = st.file_uploader(
                "Choose CSV file(s) for Sleeper 2",
                type="csv",
                key="right_csv",
                label_visibility="collapsed",
                accept_multiple_files=True
            )
            
            if right_files:
                try:
                    # Load and process the CSV files
                    merged_data, statistics, filename, filenames = csv_loader.load_csv_files(right_files)
                    
                    # Store in session state
                    csv_loader.store_csv_data_in_session(
                        merged_data, 
                        statistics, 
                        filename, 
                        filenames, 
                        side_key="sleeper_2"
                    )
                    
                    # Downsample and apply to curve editor
                    downsampled = dp.downsample_pressure_map(merged_data, target_shape=(17, 9))
                    dp.apply_pressure_map_to_curve(downsampled, side_key="sleeper_2", num_control_points=6)
                    
                    st.rerun()
                                
                except Exception as e:
                    st.error(f"Error loading files: {e}")
    else:
        st.subheader("Add a second sleeper to upload CSV files")

st.markdown("---")



def go_prev():
    st.switch_page("pages/1_Profile.py")

def go_next():
    st.switch_page("pages/3_Configure.py")

ui.nav_row(left_label="PREVIOUS", left_action=go_prev, right_label="NEXT", right_action=go_next)
