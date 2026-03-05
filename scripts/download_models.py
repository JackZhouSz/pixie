#!/usr/bin/env python3
"""
Download model-related directories from a Hugging Face dataset repo.
"""

import argparse
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


def download_models(
    dataset_repo: str = "vlongle/pixie",
    download_dirs: list[str] | None = None,
    force_download: bool = False,
    local_dir: str | None = None,
) -> None:
    project_root = Path(__file__).resolve().parent.parent
    download_path = Path(local_dir) if local_dir else project_root
    download_path.mkdir(parents=True, exist_ok=True)

    repo_files = list_repo_files(repo_id=dataset_repo, repo_type="dataset")
    data_files = [f for f in repo_files if f != "README.md" and not f.startswith(".")]
    available_dirs = sorted({f.split("/")[0] for f in data_files if "/" in f})

    if download_dirs:
        dirs_to_download = [d for d in download_dirs if d in available_dirs]
        missing = sorted(set(download_dirs) - set(dirs_to_download))
        if missing:
            print(f"Requested directories not found: {missing}")
    else:
        dirs_to_download = available_dirs

    if not dirs_to_download:
        print("No directories selected for download.")
        return

    allow_patterns = []
    for dir_name in dirs_to_download:
        allow_patterns.extend([f"{dir_name}/*", f"{dir_name}/**/*"])

    snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        local_dir=download_path,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        ignore_patterns=["README.md", ".gitattributes"],
        force_download=force_download,
    )

    print(f"Download complete to: {download_path}")
    print(f"Downloaded directories: {dirs_to_download}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Pixie model artifacts from Hugging Face.")
    parser.add_argument("--dataset-repo", default="vlongle/pixie", help="Hugging Face dataset repo id.")
    parser.add_argument("--dirs", nargs="*", help="Specific directories to download (default: all).")
    parser.add_argument("--local-dir", help="Local directory to download into (default: project root).")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    args = parser.parse_args()

    download_models(
        dataset_repo=args.dataset_repo,
        download_dirs=args.dirs,
        force_download=args.force,
        local_dir=args.local_dir,
    )


if __name__ == "__main__":
    main()
