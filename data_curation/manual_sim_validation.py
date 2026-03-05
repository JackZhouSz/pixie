import streamlit as st
import os
import math
import random
import hydra
from omegaconf import DictConfig
from pixie.utils import resolve_paths, load_json, save_json, set_logger

# Set page configuration for wider layout
st.set_page_config(layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stCheckbox { padding: 0 !important; margin: 0 !important; min-height: 20px !important; }
    .stCheckbox > label { padding: 0 !important; }
    .stCheckbox > label > p { font-size: 0.7rem !important; }
    .video-grid { margin: 2px; border: 1px solid #eee; padding: 2px; }
    .main .block-container { max-width: 100% !important; padding-left: 1rem !important; padding-right: 1rem !important; }
    .stVideo > video { width: 100% !important; }
    .row-widget > div { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

def find_video_path(obj_id, save_folder, sample_id, grid_size):
    """
    Find the simulation video for a given object ID.
    Expected path: render_outputs/{obj_id}/sample_{sample_id}/gs_sim_gridsize_{grid_size}_output/output.mp4
    """
    expected_path = os.path.join(
        save_folder,
        obj_id, 
        f"sample_{sample_id}",
                f"gs_sim_gridsize_{grid_size}_output", 
        "output.mp4"
    )
    if os.path.exists(expected_path):
        return expected_path
    return None

def display_video_grid(items, flip_label, flip_prefix, columns_per_row, save_folder, sample_id, grid_size):
    """
    Display a grid of videos with checkboxes
    """
    n_rows = math.ceil(len(items) / columns_per_row)
    
    for row_idx in range(n_rows):
        cols = st.columns(columns_per_row, gap="small")
        start_idx = row_idx * columns_per_row
        end_idx = min(start_idx + columns_per_row, len(items))
        row_items = items[start_idx:end_idx]
        
        for col_idx, (obj_id, data, _) in enumerate(row_items):
            if col_idx < len(cols):
                with cols[col_idx]:
                    video_path = find_video_path(obj_id, save_folder, sample_id, grid_size)
                    
                    st.markdown(f'<div class="video-grid">', unsafe_allow_html=True)
                    
                    if video_path:
                        st.video(video_path, autoplay=True, muted=True, loop=True)
                        st.caption(f"ID: {obj_id[:8]}...")
                        
                        # Checkbox to flip status
                        flip_key = f"{flip_prefix}_{obj_id}"
                        st.checkbox(flip_label, key=flip_key)
                        
                        with st.expander("Details"):
                            st.code(obj_id, language="text")
                            st.text(f"Path: {video_path}")
                    else:
                        st.warning(f"Video not found for {obj_id[:8]}...")
                        st.caption(f"ID: {obj_id}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    set_logger()
    cfg = resolve_paths(cfg)

    validation_cfg = cfg.data_curation.manual_sim_validation
    assert validation_cfg.obj_class, "obj_class must be specified for manual simulation validation"
    assert validation_cfg.num_samples >= -1, "num_samples must be -1 or a non-negative integer"

    obj_class = validation_cfg.obj_class
    num_samples = validation_cfg.num_samples
    save_folder = validation_cfg.render_outputs_dir or cfg.paths.render_outputs_dir
    sample_id = validation_cfg.sample_id
    grid_size = validation_cfg.grid_size
    columns_per_row = validation_cfg.columns_per_row
    json_path = os.path.join(
        cfg.paths.vlm_filtering_results_dir,
        obj_class,
        validation_cfg.input_file,
    )
    new_json_path = os.path.join(
        cfg.paths.vlm_filtering_results_dir,
        obj_class,
        validation_cfg.output_file,
    )

    st.title(f"Simulation Quality Validation: {obj_class}")

    if not os.path.exists(json_path):
        st.error(f"JSON file not found at {json_path}")
        return

    # Load data
    # Structure: { "obj_id": {"is_appropriate": bool, ...}, ... }
    all_data = load_json(json_path)
    
    # We only care about objects that were previously marked as appropriate
    # AND have a generated video.
    
    valid_items = []
    missing_video_items = []
    
    for tag, data in all_data.items():
        if not data.get("is_appropriate"):
            continue
            
        obj_id = tag.split("/")[-1]
        video_path = find_video_path(obj_id, save_folder, sample_id, grid_size)
        if video_path:
            # Check if we already have a validation status for this object
            # If not, default to True (appropriate)
            if "is_simulation_valid" not in data:
                data["is_simulation_valid"] = True
            valid_items.append((obj_id, data, tag))
        else:
            missing_video_items.append(obj_id)

    if num_samples != -1:
        assert num_samples >= 0, "--num_samples must be -1 or non-negative"
        if num_samples < len(valid_items):
            valid_items = random.sample(valid_items, num_samples)

    # Separate into valid vs invalid simulations based on current status
    good_sims = [(oid, d, t) for oid, d, t in valid_items if d.get("is_simulation_valid", True)]
    bad_sims = [(oid, d, t) for oid, d, t in valid_items if not d.get("is_simulation_valid", True)]

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Candidates", len(valid_items))
    with col2: st.metric("Good Simulations", len(good_sims))
    with col3: st.metric("Bad Simulations", len(bad_sims))
    with col4: st.metric("Missing Videos", len(missing_video_items))

    with st.form("validation_form"):
        st.subheader(f"Valid Simulations ({len(good_sims)})")
        st.markdown("Check box to mark as **INVALID** (bad physics, artifacts, etc.)")
        display_video_grid(
            good_sims,
            "Mark as Invalid",
            "flip_to_invalid",
            columns_per_row,
            save_folder,
            sample_id,
            grid_size,
        )
        
        st.markdown("---")
        
        st.subheader(f"Invalid Simulations ({len(bad_sims)})")
        st.markdown("Check box to mark as **VALID** (restore to dataset)")
        display_video_grid(
            bad_sims,
            "Mark as Valid",
            "flip_to_valid",
            columns_per_row,
            save_folder,
            sample_id,
            grid_size,
        )
        
        st.markdown("---")
        submitted = st.form_submit_button("Save Validation Results", use_container_width=True)
        
        if submitted:
            changes = 0
            
            # Process flips from Good -> Bad
            for obj_id, data, tag in good_sims:
                key = f"flip_to_invalid_{obj_id}"
                if st.session_state.get(key, False):
                    all_data[tag]["is_simulation_valid"] = False
                    changes += 1
            
            # Process flips from Bad -> Good
            for obj_id, data, tag in bad_sims:
                key = f"flip_to_valid_{obj_id}"
                if st.session_state.get(key, False):
                    all_data[tag]["is_simulation_valid"] = True
                    changes += 1
            
            if changes > 0:
                save_json(all_data, new_json_path)
                st.success(f"Saved {changes} changes to {new_json_path}")
                # Also update the original file to persist state? 
                # Uncomment if you want to overwrite the input file
                # save_json(all_data, json_path)
            else:
                st.info("No changes made.")

if __name__ == "__main__":
    main()
