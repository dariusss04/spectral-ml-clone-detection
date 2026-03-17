# Build clone-search configuration JSON for benchmarking splits.

import json
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "meta" / "programs_metadata.txt"))
SPLITS_DIR = Path(os.getenv("PCD_SPLITS_DIR", DATA_DIR / "splits"))
TEST_SPLIT_DIR = Path(os.getenv("PCD_TEST_SPLIT_DIR", SPLITS_DIR / "test"))
OUTPUT_FILE = Path(os.getenv("PCD_CLONE_SEARCH_CONFIG", DATA_DIR / "meta" / "clone_search_config.json"))

RANDOM_SEED = int(os.getenv("PCD_RANDOM_SEED", "42"))
TARGET_FRACTION = float(os.getenv("PCD_TARGET_FRACTION", "0.1"))


def parse_samples(samples_file_path):
    sample_dict = {}
    with open(samples_file_path, "r") as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        idx = int(lines[i].strip())
        program_info = lines[i + 1].strip()
        sample_dict[idx] = program_info
    return sample_dict


def filter_programs_by_type(samples_dict, split_folder, search_type, version):
    available_programs = {
        int(f.replace("program", "").replace(".pt", ""))
        for f in os.listdir(split_folder) if f.endswith(".pt")
    }

    filtered_indices = []
    for idx, info in samples_dict.items():
        if idx not in available_programs:
            continue

        parts = info.split("_")

        if search_type == "o" and version == parts[4]:
            filtered_indices.append(idx)
        elif search_type == "c":
            prog_compiler_full = parts[1]
            if "-" in version:
                user_compiler, user_ver = version.split("-", 1)
                user_ver = user_ver.split(".")[0]
                user_normalized = f"{user_compiler}-{user_ver}"

                if "-" in prog_compiler_full:
                    actual_compiler, actual_ver = prog_compiler_full.split("-", 1)
                    actual_ver = actual_ver.split(".")[0]
                    actual_normalized = f"{actual_compiler}-{actual_ver}"
                else:
                    actual_normalized = prog_compiler_full

                if user_normalized == actual_normalized:
                    filtered_indices.append(idx)
            else:
                user_compiler_name = version
                actual_compiler_name = prog_compiler_full.split("-")[0]
                if actual_compiler_name == user_compiler_name:
                    filtered_indices.append(idx)
        elif search_type == "a" and version == parts[2]:
            filtered_indices.append(idx)
        elif search_type == "b" and version == parts[3]:
            filtered_indices.append(idx)

    return filtered_indices


def get_subset(indices, fraction):
    if not indices:
        return []
    subset_size = max(1, int(fraction * len(indices)))
    return random.sample(indices, subset_size)


def main():
    random.seed(RANDOM_SEED)

    samples_dict = parse_samples(SAMPLES_FILE)

    all_configs = [
        ("o", "O0", "O1"),
        ("o", "O0", "O2"),
        ("o", "O0", "O3"),
        ("o", "O1", "O2"),
        ("o", "O1", "O3"),
        ("o", "O2", "O3"),
        ("c", "gcc", "clang"),
        ("c", "gcc-4", "gcc-8"),
        ("c", "clang-4", "clang-7"),
        ("a", "arm", "mips"),
        ("a", "arm", "x86"),
        ("a", "mips", "x86"),
        ("b", "32", "64"),
    ]

    cs_data = {}
    for stype, v1, v2 in all_configs:
        targets = filter_programs_by_type(samples_dict, TEST_SPLIT_DIR, stype, v1)
        repositories = filter_programs_by_type(samples_dict, TEST_SPLIT_DIR, stype, v2)

        target_subset = get_subset(targets, TARGET_FRACTION)
        config_key = f"{stype}_{v1}_vs_{v2}"

        cs_data[config_key] = {
            "search_type": stype,
            "version1": v1,
            "version2": v2,
            "target": target_subset,
            "repository": repositories,
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(cs_data, f, indent=2)

    print(f"[INFO] Saved clone search sets for {len(all_configs)} configs to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
