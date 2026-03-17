# Summarize unique binaries per project from the metadata file.

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "meta" / "programs_metadata.txt"))


def count_unique_binaries(samples_file):
    project_binaries = {}

    with open(samples_file, "r") as f:
        lines = f.read().splitlines()

    for i in range(0, len(lines), 2):
        program_info = lines[i + 1]
        parts = program_info.split("_")
        if not parts:
            continue

        project_name = parts[0].split("-")[0]
        binary_name = parts[-1]
        project_binaries.setdefault(project_name, set()).add(binary_name)

    return {project: len(binaries) for project, binaries in project_binaries.items()}


def main():
    if not SAMPLES_FILE.is_file():
        raise FileNotFoundError(f"Samples file not found: {SAMPLES_FILE}")

    counts = count_unique_binaries(SAMPLES_FILE)
    for project in sorted(counts.keys()):
        print(f"{project}: {counts[project]}")


if __name__ == "__main__":
    main()
