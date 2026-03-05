# PixieVerse Data README

This file documents the dataset directory rooted at `paths.base_path` (Hydra config).
By default, `paths.base_path` resolves to the current working directory when unset.

## Hugging Face

- Dataset (PixieVerse): [https://huggingface.co/datasets/vlongle/pixieverse](https://huggingface.co/datasets/vlongle/pixieverse)
- Models/checkpoints: [https://huggingface.co/datasets/vlongle/pixie](https://huggingface.co/datasets/vlongle/pixie)

Download PixieVerse archives:

```bash
python scripts/download_data.py \
  --dataset-repo vlongle/pixieverse \
  --dirs archives \
  --local-dir /path/to/pixieverse_root
```

Download only one class archive for testing:

```bash
python scripts/download_data.py \
  --dataset-repo vlongle/pixieverse \
  --dirs archives \
  --obj-class tree \
  --local-dir /path/to/pixieverse_root
```

Unpack into the standard folder layout:

```bash
ROOT=/path/to/pixieverse_root
set -euo pipefail

for d in data outputs render_outputs vlm_seg_results vlm_seg_critic_results vlm_seg_mat_sample_results; do
  src="$ROOT/archives/$d"
  dst="$ROOT/$d"
  mkdir -p "$dst"
  [ -d "$src" ] || { echo "[skip] $src not found"; continue; }
  echo "[dir] $d"
  for a in "$src"/*.tar "$src"/*.tar.gz; do
    [ -e "$a" ] || continue
    echo "  -> extracting $(basename "$a")"
    tar -xf "$a" -C "$dst" --checkpoint=2000 --checkpoint-action=echo="    ... extracted 2000 more entries"
    echo "  <- done $(basename "$a")"
  done
done
```

## Top-level folders

- `data`
- `outputs`
- `render_outputs`
- `vlm_seg_results`
- `vlm_seg_critic_results`
- `vlm_seg_mat_sample_results`

## How data is generated

Class-level launcher:

```bash
python generate_slurm_vlm_job.py \
  --obj_ids_json vlm_data_filtering_results/<obj_class>/all_results_corrected.json \
  --obj_class <obj_class> \
  --overwrite_sim \
  --overwrite_vlm \
  --submit \
  --gray_threshold 0.05 \
  --qos ee-high \
  --partition eaton-compute
```

Per-object pipeline:

```bash
python run_seg.py \
  --obj_id <obj_id> \
  --obj_class <obj_class> \
  --num_alternative_queries 5 \
  --num_sample_mat 1 \
  --gray_threshold 0.05
```

## Folder details

### `data/`

Per-object image data and assets used by reconstruction/training.

```text
data/
  <obj_id>/
    train/
      0001.png
      ...
```

### `outputs/`

Intermediate reconstruction/training outputs (organized by object/method/run).

```text
outputs/
  <obj_id>/
    <method>/
      <run_id_or_timestamp>/
        ...
```

### `render_outputs/`

Final per-object simulation/render artifacts used for curation and validation.

Simulation folder naming:

- Current: `gs_sim_gridsize_<D>_output`
- Old: `gs_sim_gridsize_<D>_neurips_paper_output` (migrated to new naming)

```text
render_outputs/
  <obj_id>/
    sample_0/
      gs_sim_gridsize_64_output/
        output.mp4
        output.gif
        ...
```

### `vlm_seg_results/`

Raw VLM segmentation stage results per object.

### `vlm_seg_critic_results/`

VLM critic outputs that evaluate segmentation candidates.

### `vlm_seg_mat_sample_results/`

Material/physics parameter sampling outputs (often per object and sample index).

```text
vlm_seg_mat_sample_results/
  <obj_id>/
    sample_0/
      chosen_vlm_results.json
      ...
```

## Manual validation

```bash
streamlit run data_curation/manual_sim_validation.py data_curation.manual_sim_validation.obj_class=<obj_class>
```

The validator uses class-level JSON under `vlm_data_filtering_results/<obj_class>/` and reads videos from:

`render_outputs/<obj_id>/sample_*/gs_sim_gridsize_<D>_output/`

## Hugging Face dataset card

The Hugging Face dataset README (`README.md`) should mirror this file (`data_readme.md`).

## Citation

If you find this dataset useful, please consider citing:

```bibtex
@article{le2025pixie,
  title={Pixie: Fast and Generalizable Supervised Learning of 3D Physics from Pixels},
  author={Le, Long and Lucas, Ryan and Wang, Chen and Chen, Chuhao and Jayaraman, Dinesh and Eaton, Eric and Liu, Lingjie},
  journal={arXiv preprint arXiv:2508.17437},
  year={2025}
}
```
