#!/usr/bin/env python3
"""
Upload model-related directories to a Hugging Face dataset repo.
"""

import argparse
import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, login


DEFAULT_UPLOAD_DIRS = [
    "checkpoints_continuous_mse",
    "checkpoints_discrete",
    "real_scene_data",
    "real_scene_models",
]

IGNORE_PATTERNS = [
    "*.pyc",
    "__pycache__",
    ".DS_Store",
    "*.tmp",
    "*.log",
    "wandb",
    ".git",
    ".gitignore",
]


def should_ignore_file(file_path: Path) -> bool:
    file_name = file_path.name
    return any(fnmatch.fnmatch(file_name, pattern) for pattern in IGNORE_PATTERNS)


def build_upload_ignore_patterns() -> list[str]:
    ignore_patterns: list[str] = []
    for pattern in IGNORE_PATTERNS:
        if pattern == "__pycache__":
            ignore_patterns.extend(["**/__pycache__/**", "__pycache__"])
        elif pattern == ".git":
            ignore_patterns.extend(["**/.git/**", ".git"])
        elif pattern == "wandb":
            ignore_patterns.extend(["**/wandb/**", "wandb"])
        else:
            ignore_patterns.append(pattern)
    return ignore_patterns


def upload_directory(api: HfApi, local_dir: Path, repo_id: str, repo_dir_name: str | None = None) -> tuple[int, int]:
    if repo_dir_name is None:
        repo_dir_name = local_dir.name

    total_files = 0
    total_size = 0
    for file_path in local_dir.rglob("*"):
        if file_path.is_file() and not should_ignore_file(file_path):
            total_files += 1
            total_size += file_path.stat().st_size

    if total_files == 0:
        print(f"Skipping {local_dir}: no files to upload.")
        return 0, 0

    print(f"Uploading {local_dir} -> {repo_dir_name}/ ({total_files} files)")
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=repo_dir_name,
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=build_upload_ignore_patterns(),
    )
    return total_files, total_size


def upload_models(dataset_repo: str, upload_dirs: list[str], token: str | None) -> None:
    if token:
        login(token=token)
    else:
        login()

    api = HfApi()
    project_root = Path(__file__).resolve().parent.parent

    total_uploaded_files = 0
    total_uploaded_size = 0
    for dir_name in upload_dirs:
        local_dir = project_root / dir_name
        if not local_dir.exists():
            print(f"Skipping missing directory: {local_dir}")
            continue
        if not local_dir.is_dir():
            print(f"Skipping non-directory path: {local_dir}")
            continue
        files, size = upload_directory(api, local_dir, dataset_repo)
        total_uploaded_files += files
        total_uploaded_size += size

    print(f"Upload complete: {total_uploaded_files} files, {total_uploaded_size / (1024 * 1024):.1f} MB")
    print(f"https://huggingface.co/datasets/{dataset_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Pixie model artifacts to Hugging Face.")
    parser.add_argument("--dataset-repo", default="vlongle/pixie", help="Hugging Face dataset repo id.")
    parser.add_argument("--dirs", nargs="*", help=f"Directories to upload (default: {DEFAULT_UPLOAD_DIRS}).")
    parser.add_argument("--token", help="Hugging Face token. If unset, interactive login is used.")
    args = parser.parse_args()

    upload_dirs = args.dirs if args.dirs else DEFAULT_UPLOAD_DIRS
    upload_models(dataset_repo=args.dataset_repo, upload_dirs=upload_dirs, token=args.token)


if __name__ == "__main__":
    main()
