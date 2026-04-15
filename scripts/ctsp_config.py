#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

# Base folders where your sample-specific files live.
ZARR_BASE_DIR = Path("/Volumes/JubayerUKD/TNHL/snakemake_outputs/zarr/stardist")
THRESHOLD_BASE_DIR = Path("/Volumes/JubayerUKD/TNHL/threshold/outputs")

# File naming rules derived from sample ID.
# Example for sample_id "1-2":
# dataset_file -> "1-2_v3.zarr"
# threshold_file -> "1-2.txt"
DATASET_FILE_TEMPLATE = "{sample_id}_v3.zarr"
THRESHOLD_FILE_TEMPLATE = "{sample_id}.txt"

# Control which samples to process when running ctsp_astir.py directly.
# Single sample example: RUN_SAMPLE_IDS = ["1-1"]
# Multiple sample example: RUN_SAMPLE_IDS = ["1-1", "1-2"]
# Auto-discover all samples from filesystem: RUN_SAMPLE_IDS = ["all"]
RUN_SAMPLE_IDS = ["1-1", "1-2", "1-3"]


def get_sample_paths(sample_id: str | None = None) -> tuple[str, str]:
    if sample_id is None:
        sample_ids = get_run_sample_ids()
        if not sample_ids:
            raise ValueError("No sample IDs available from RUN_SAMPLE_IDS.")
        sample_key = sample_ids[0]
    else:
        sample_key = sample_id

    dataset_file = DATASET_FILE_TEMPLATE.format(sample_id=sample_key)
    threshold_file = THRESHOLD_FILE_TEMPLATE.format(sample_id=sample_key)
    dataset_path = (ZARR_BASE_DIR / dataset_file).as_posix()
    threshold_path = (THRESHOLD_BASE_DIR / threshold_file).as_posix()
    return dataset_path, threshold_path


def _discover_sample_ids() -> list[str]:
    threshold_ids = {path.stem for path in THRESHOLD_BASE_DIR.glob("*.txt")}
    dataset_ids = {
        path.name[: -len("_v3.zarr")]
        for path in ZARR_BASE_DIR.glob("*_v3.zarr")
        if path.name.endswith("_v3.zarr")
    }
    return sorted(threshold_ids.intersection(dataset_ids))


def get_run_sample_ids() -> list[str]:
    if not RUN_SAMPLE_IDS:
        raise ValueError("RUN_SAMPLE_IDS is empty. Add at least one sample ID.")

    if len(RUN_SAMPLE_IDS) == 1 and RUN_SAMPLE_IDS[0].lower() == "all":
        discovered_ids = _discover_sample_ids()
        if not discovered_ids:
            raise ValueError(
                "RUN_SAMPLE_IDS=['all'] but no matching samples were discovered. "
                "Expected pairs: '<id>_v3.zarr' in ZARR_BASE_DIR and '<id>.txt' in THRESHOLD_BASE_DIR."
            )
        return discovered_ids

    cleaned_ids = [sample_id.strip() for sample_id in RUN_SAMPLE_IDS if sample_id and sample_id.strip()]
    if not cleaned_ids:
        raise ValueError("RUN_SAMPLE_IDS contains only empty values.")

    return cleaned_ids
