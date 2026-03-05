#!/usr/bin/env python3
"""
Upload PixieVerse data (survivor objects only) to Hugging Face datasets.

Supports two modes:
- raw: upload selected object folders directly (many API operations, slower)
- archive: pack per-(dir,class) tar archives, then upload archives (faster, better visibility)
"""

import argparse
import json
import errno
import os
import sys
import time
import threading
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, login
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError


DEFAULT_CLASSES = [
    "tree",
    "flowers",
    "rubber_ducks_and_toys",
    "sport_balls",
    "sand",
    "snow_and_mud",
    "grass",
    "shrubs",
    "metal_crates",
    "soda_cans",
]

DEFAULT_DATA_DIRS = [
    "vlm_seg_results", 
    "vlm_seg_critic_results",
    "vlm_seg_mat_sample_results",
    "data",
    "outputs",
    "render_outputs",
]


ARCHIVE_EXTENSIONS = {
    "none": ".tar",
    "gz": ".tar.gz",
}


def has_feature_artifact(obj_dir: Path) -> bool:
    return (
        (obj_dir / "clip_features_features.npy").exists()
        or (obj_dir / "clip_features.npz").exists()
    )


class Spinner:
    def __init__(self, message: str, interval_s: float = 0.2):
        self.message = message
        self.interval_s = interval_s
        self._running = False
        self._thread = None

    def _spin(self) -> None:
        frames = "|/-\\"
        i = 0
        while self._running:
            sys.stdout.write(f"\r{self.message} {frames[i % len(frames)]}")
            sys.stdout.flush()
            i += 1
            time.sleep(self.interval_s)
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._running = False
        if self._thread is not None:
            self._thread.join()


def summarize_obj_dirs(source_dir: Path, object_ids: list[str]) -> tuple[int, int, int]:
    """Return (object_dir_count, file_count, total_bytes) for selected object ids."""
    obj_count = 0
    file_count = 0
    total_bytes = 0
    for obj_id in object_ids:
        obj_dir = source_dir / obj_id
        if not obj_dir.is_dir():
            continue
        obj_count += 1
        for root, _, files in os.walk(obj_dir):
            file_count += len(files)
            for fname in files:
                fp = Path(root) / fname
                if fp.exists():
                    total_bytes += fp.stat().st_size
    return obj_count, file_count, total_bytes


def iter_obj_files(source_dir: Path, object_ids: list[str]):
    """Yield tuples (absolute_file_path, archive_relative_path)."""
    for obj_id in object_ids:
        obj_dir = source_dir / obj_id
        if not obj_dir.is_dir():
            continue
        for root, _, files in os.walk(obj_dir):
            root_path = Path(root)
            rel_root = root_path.relative_to(obj_dir)
            for fname in files:
                abs_fp = root_path / fname
                rel_fp = Path(obj_id) / rel_root / fname
                yield abs_fp, rel_fp


def parse_classes(classes_arg: str | None) -> list[str]:
    if not classes_arg:
        return DEFAULT_CLASSES
    classes = [c.strip() for c in classes_arg.split(",") if c.strip()]
    assert classes, "classes must not be empty"
    return classes


def collect_survivor_ids(
    classes: list[str],
    results_root: Path,
    render_outputs_dir: Path,
    sample_id: int,
    grid_size: int,
) -> tuple[set[str], dict[str, dict[str, int]], dict[str, set[str]]]:
    survivor_ids: set[str] = set()
    stats: dict[str, dict[str, int]] = {}
    survivor_ids_by_class: dict[str, set[str]] = {obj_class: set() for obj_class in classes}

    for obj_class in classes:
        validated_path = results_root / obj_class / "all_results_validated.json"
        corrected_path = results_root / obj_class / "all_results_corrected.json"
        if validated_path.exists():
            json_path = validated_path
            assume_zero_filtered = False
        elif corrected_path.exists():
            json_path = corrected_path
            assume_zero_filtered = True
            print(
                f"Warning: can't find {validated_path}. "
                f"Assuming 0 objects were manually filtered for class '{obj_class}'."
            )
        else:
            print(
                f"Warning: can't find both validated and corrected files for class '{obj_class}'. "
                "Skipping this class."
            )
            stats[obj_class] = {
                "appropriate": 0,
                "with_sim_output": 0,
                "survived": 0,
            }
            continue

        with open(json_path, "r") as f:
            results = json.load(f)

        appropriate = 0
        with_sim_output = 0
        survived = 0
        for tag, meta in results.items():
            if meta.get("is_appropriate") is not True:
                continue
            appropriate += 1

            obj_id = tag.split("/")[-1]
            obj_dir = render_outputs_dir / obj_id
            has_feat = has_feature_artifact(obj_dir)
            mat_fp = render_outputs_dir / obj_id / f"sample_{sample_id}" / "material_grid.npy"
            sim_dir = render_outputs_dir / obj_id / f"sample_{sample_id}" / f"gs_sim_gridsize_{grid_size}_output"
            mp4_fp = sim_dir / "output.mp4"

            has_sim = has_feat and mat_fp.exists() and sim_dir.is_dir() and mp4_fp.exists()
            if has_sim:
                with_sim_output += 1

            is_simulation_valid = meta.get("is_simulation_valid", True) if assume_zero_filtered else (meta.get("is_simulation_valid") is True)
            if has_sim and is_simulation_valid:
                survived += 1
                survivor_ids.add(obj_id)
                survivor_ids_by_class[obj_class].add(obj_id)

        stats[obj_class] = {
            "appropriate": appropriate,
            "with_sim_output": with_sim_output,
            "survived": survived,
        }
    return survivor_ids, stats, survivor_ids_by_class


def upload_selected_ids(
    api: HfApi,
    repo_id: str,
    dataset_root: Path,
    include_dirs: list[str],
    object_ids: set[str],
) -> None:
    ignore_patterns = [".gitattributes", "*.pyc", "__pycache__", ".DS_Store"]
    sorted_ids = sorted(object_ids)
    total_dirs = len(include_dirs)

    for idx, dir_name in enumerate(include_dirs, start=1):
        source_dir = dataset_root / dir_name
        if not source_dir.is_dir():
            print(f"[{idx}/{total_dirs}] Skipping missing source directory: {source_dir}")
            continue

        allow_patterns = [f"{obj_id}/**" for obj_id in sorted_ids if (source_dir / obj_id).exists()]
        if not allow_patterns:
            print(f"[{idx}/{total_dirs}] Skipping {dir_name}/ (no matching object folders).")
            continue

        object_dirs_found, total_files, total_bytes = summarize_obj_dirs(source_dir, sorted_ids)
        size_gb = total_bytes / (1024 ** 3)
        print(
            f"[{idx}/{total_dirs}] {dir_name}/ -> {object_dirs_found} object folders, "
            f"{total_files} files, {size_gb:.2f} GB"
        )

        start = time.time()
        with Spinner(f"[{idx}/{total_dirs}] Uploading {dir_name}/"):
            api.upload_folder(
                folder_path=str(source_dir),
                path_in_repo=dir_name,
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        elapsed = time.time() - start
        print(f"[{idx}/{total_dirs}] Done {dir_name}/ in {elapsed:.1f}s")


def create_archive_for_class(
    source_dir: Path,
    object_ids: list[str],
    archive_path: Path,
    compression: str,
) -> tuple[int, int]:
    """
    Build an archive for selected object ids under one top-level directory.
    Returns (files_written, bytes_written).
    """
    assert compression in ARCHIVE_EXTENSIONS, f"Unsupported compression: {compression}"
    mode = "w" if compression == "none" else "w:gz"

    if archive_path.exists():
        existing_size_gb = archive_path.stat().st_size / (1024 ** 3)
        print(f"      Reusing existing archive {archive_path.name} ({existing_size_gb:.2f} GB)")
        return 0, 0

    _, total_files, total_bytes = summarize_obj_dirs(source_dir, object_ids)
    print(
        f"      Packing {source_dir.name}: {len(object_ids)} obj ids, "
        f"{total_files} files, {total_bytes / (1024 ** 3):.2f} GB"
    )
    if total_files == 0:
        return 0, 0

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    files_written = 0
    bytes_written = 0
    start = time.time()
    last_log = start

    try:
        with tarfile.open(archive_path, mode=mode) as tar:
            for abs_fp, rel_fp in iter_obj_files(source_dir, object_ids):
                if not abs_fp.exists():
                    continue
                tar.add(abs_fp, arcname=str(rel_fp), recursive=False)
                files_written += 1
                bytes_written += abs_fp.stat().st_size

                now = time.time()
                if files_written % 2000 == 0 or now - last_log > 10:
                    pct = 100.0 * files_written / total_files
                    print(
                        f"        [{source_dir.name}] packed {files_written}/{total_files} files "
                        f"({pct:.1f}%), {bytes_written / (1024 ** 3):.2f} GB"
                    )
                    last_log = now
    except OSError as e:
        if archive_path.exists():
            archive_path.unlink()
            print(f"      Removed partial archive after failure: {archive_path}")
        if e.errno == errno.ENOSPC:
            raise RuntimeError(
                f"No space left while packing {archive_path}. "
                "Use a larger --archive-tmp-dir (e.g., your mounted dataset path)."
            ) from e
        raise

    elapsed = time.time() - start
    archive_size_gb = archive_path.stat().st_size / (1024 ** 3)
    print(
        f"      Packed archive {archive_path.name} in {elapsed:.1f}s "
        f"(archive size: {archive_size_gb:.2f} GB)"
    )
    return files_written, bytes_written


def upload_archives_by_class(
    api: HfApi,
    repo_id: str,
    dataset_root: Path,
    include_dirs: list[str],
    classes: list[str],
    survivor_ids_by_class: dict[str, set[str]],
    archive_tmp_dir: Path,
    compression: str,
    keep_archives: bool,
    force_repack: bool,
) -> None:
    ext = ARCHIVE_EXTENSIONS[compression]
    jobs = []
    for dir_name in include_dirs:
        source_dir = dataset_root / dir_name
        if not source_dir.is_dir():
            continue
        for obj_class in classes:
            obj_ids = sorted(survivor_ids_by_class.get(obj_class, set()))
            if obj_ids:
                jobs.append((dir_name, source_dir, obj_class, obj_ids))

    total_jobs = len(jobs)
    if total_jobs == 0:
        print("No archive upload jobs to run.")
        return

    print(f"Archive mode: {total_jobs} upload jobs")
    archive_relpaths: list[str] = []
    archive_abspaths: list[Path] = []

    for idx, (dir_name, source_dir, obj_class, obj_ids) in enumerate(jobs, start=1):
        print(f"[{idx}/{total_jobs}] {dir_name}/{obj_class}: start")
        archive_name = f"{obj_class}{ext}"
        archive_path = archive_tmp_dir / dir_name / archive_name

        if force_repack and archive_path.exists():
            archive_path.unlink()
            print(f"      Removed existing archive due to --force-repack: {archive_path}")

        create_archive_for_class(
            source_dir=source_dir,
            object_ids=obj_ids,
            archive_path=archive_path,
            compression=compression,
        )
        if not archive_path.exists():
            print(f"[{idx}/{total_jobs}] {dir_name}/{obj_class}: nothing to upload")
            continue

        rel = str(archive_path.relative_to(archive_tmp_dir))
        archive_relpaths.append(rel)
        archive_abspaths.append(archive_path)
        print(f"[{idx}/{total_jobs}] Queued for upload: archives/{rel}")

    if not archive_relpaths:
        print("No archives to upload after packing/reuse.")
        return

    print(
        "Packing/reuse phase done. "
        f"Uploading {len(archive_relpaths)} queued archives with per-archive commits "
        "(compatible with current huggingface_hub version)."
    )
    total_upload = len(archive_abspaths)
    for i, archive_path in enumerate(archive_abspaths, start=1):
        rel = str(archive_path.relative_to(archive_tmp_dir))
        remote_path = f"archives/{rel}"
        size_gb = archive_path.stat().st_size / (1024 ** 3)
        print(f"[upload {i}/{total_upload}] {remote_path} ({size_gb:.2f} GB)")
        start = time.time()
        with Spinner(f"[upload {i}/{total_upload}] Uploading {remote_path}"):
            api.upload_file(
                path_or_fileobj=str(archive_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
        elapsed = time.time() - start
        print(f"[upload {i}/{total_upload}] Done {remote_path} in {elapsed:.1f}s")
    print("Archive upload complete.")

    if not keep_archives:
        for archive_path in archive_abspaths:
            if archive_path.exists():
                archive_path.unlink()
        print(f"Removed {len(archive_abspaths)} local archives from {archive_tmp_dir}")


def upload_data(
    dataset_repo: str,
    dataset_root: Path,
    results_root: Path,
    classes: list[str],
    include_dirs: list[str],
    sample_id: int,
    grid_size: int,
    token: str | None,
    mode: str,
    archive_tmp_dir: Path,
    archive_compression: str,
    keep_archives: bool,
    force_repack: bool,
    auto_create_repo: bool,
    private_repo: bool,
) -> None:
    assert mode in {"raw", "archive"}, f"Unsupported mode: {mode}"
    assert archive_compression in ARCHIVE_EXTENSIONS, f"Unsupported archive compression: {archive_compression}"

    if token:
        login(token=token, add_to_git_credential=False)
    else:
        login(add_to_git_credential=False)

    # Faster transfer path when available.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print(f"HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info.get("name", "unknown")
        print(f"Authenticated as: {username}")
    except HfHubHTTPError as e:
        raise RuntimeError(
            "Failed to authenticate with Hugging Face Hub. "
            "Please verify your token has write permission."
        ) from e

    try:
        api.repo_info(repo_id=dataset_repo, repo_type="dataset")
        print(f"Dataset repo found: {dataset_repo}")
    except RepositoryNotFoundError:
        if not auto_create_repo:
            raise RuntimeError(
                f"Dataset repo not found: {dataset_repo}. "
                "Create it on Hugging Face first, or run without --no-auto-create-repo."
            )
        print(f"Dataset repo not found. Creating: {dataset_repo}")
        api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=private_repo, exist_ok=True)
        print(f"Dataset repo ready: {dataset_repo}")
    except HfHubHTTPError as e:
        raise RuntimeError(
            f"Unable to access dataset repo '{dataset_repo}'. "
            "This is usually a permission/token-scope issue."
        ) from e

    render_outputs_dir = dataset_root / "render_outputs"
    assert render_outputs_dir.is_dir(), f"Missing render_outputs directory: {render_outputs_dir}"

    survivor_ids, stats, survivor_ids_by_class = collect_survivor_ids(
        classes=classes,
        results_root=results_root,
        render_outputs_dir=render_outputs_dir,
        sample_id=sample_id,
        grid_size=grid_size,
    )

    total_appropriate = sum(v["appropriate"] for v in stats.values())
    total_sim = sum(v["with_sim_output"] for v in stats.values())
    total_survived = sum(v["survived"] for v in stats.values())
    print(f"Classes: {classes}")
    print(f"Appropriate: {total_appropriate}")
    print(f"With sim output: {total_sim}")
    print(f"Survived: {total_survived}")
    print(f"Unique survivor object ids: {len(survivor_ids)}")
    for obj_class in classes:
        class_stats = stats.get(obj_class, {"appropriate": 0, "with_sim_output": 0, "survived": 0})
        print(
            f"  - {obj_class}: app={class_stats['appropriate']}, "
            f"sim={class_stats['with_sim_output']}, survived={class_stats['survived']}"
        )

    if mode == "raw":
        upload_selected_ids(
            api=api,
            repo_id=dataset_repo,
            dataset_root=dataset_root,
            include_dirs=include_dirs,
            object_ids=survivor_ids,
        )
    else:
        upload_archives_by_class(
            api=api,
            repo_id=dataset_repo,
            dataset_root=dataset_root,
            include_dirs=include_dirs,
            classes=classes,
            survivor_ids_by_class=survivor_ids_by_class,
            archive_tmp_dir=archive_tmp_dir,
            compression=archive_compression,
            keep_archives=keep_archives,
            force_repack=force_repack,
        )

    # Make dataset card consistent with local data README.
    project_root = Path(__file__).resolve().parent.parent
    readme_path = project_root / "data_readme.md"
    assert readme_path.exists(), f"Missing {readme_path}"
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=dataset_repo,
        repo_type="dataset",
    )
    print(f"Uploaded README.md from {readme_path}")
    print(f"https://huggingface.co/datasets/{dataset_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload PixieVerse survivor data to Hugging Face.")
    parser.add_argument("--dataset-repo", default="vlongle/pixieverse", help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--dataset-root",
        default="/mnt/kostas-graid/datasets/vlongle/diffphys3d",
        help="Local dataset root directory.",
    )
    parser.add_argument(
        "--results-root",
        default="/home/vlongle/code/diffPhys3d/vlm_data_filtering_results",
        help="Root containing <class>/all_results_validated.json.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class list. Defaults to PixieVerse 10 classes (excluding jello_block).",
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=DEFAULT_DATA_DIRS,
        help=f"Top-level dataset folders to upload (default: {DEFAULT_DATA_DIRS})",
    )
    parser.add_argument("--sample-id", type=int, default=0, help="Simulation sample id.")
    parser.add_argument("--grid-size", type=int, default=64, help="Simulation grid size D.")
    parser.add_argument("--token", default=None, help="Hugging Face token.")
    parser.add_argument(
        "--mode",
        choices=["raw", "archive"],
        default="archive",
        help="Upload mode. archive is faster and more transparent for large datasets.",
    )
    parser.add_argument(
        "--archive-tmp-dir",
        default="/mnt/kostas-graid/datasets/vlongle/pixieverse/_archive_staging",
        help="Staging directory for archive mode (should have enough free space).",
    )
    parser.add_argument(
        "--archive-compression",
        choices=["none", "gz"],
        default="none",
        help="Compression in archive mode. none is fastest CPU path.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep local archives after upload.",
    )
    parser.add_argument(
        "--force-repack",
        action="store_true",
        help="Force rebuilding archives even if they already exist locally.",
    )
    parser.add_argument(
        "--no-auto-create-repo",
        action="store_true",
        help="Disable automatic dataset repo creation when missing.",
    )
    parser.add_argument(
        "--private-repo",
        action="store_true",
        help="Create dataset repo as private if auto-created.",
    )
    args = parser.parse_args()

    classes = parse_classes(args.classes)
    upload_data(
        dataset_repo=args.dataset_repo,
        dataset_root=Path(args.dataset_root),
        results_root=Path(args.results_root),
        classes=classes,
        include_dirs=args.dirs,
        sample_id=args.sample_id,
        grid_size=args.grid_size,
        token=args.token,
        mode=args.mode,
        archive_tmp_dir=Path(args.archive_tmp_dir),
        archive_compression=args.archive_compression,
        keep_archives=args.keep_archives,
        force_repack=args.force_repack,
        auto_create_repo=not args.no_auto_create_repo,
        private_repo=args.private_repo,
    )


if __name__ == "__main__":
    main()
