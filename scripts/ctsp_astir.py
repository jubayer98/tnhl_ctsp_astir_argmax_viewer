#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import spatialproteomics
import xarray as xr

from ctsp_config import get_run_sample_ids, get_sample_paths
from ctsp_html_common import generate_html_report
from report_index import update_index_html

CHANNELS = [
    "PAX5",
    "CD3",
    "CD11b",
    "CD11c",
    "CD68",
    "CD90",
    "Podoplanin",
    "CD31",
    "CD34",
    "CD56",
    "CD57",
    "CD138",
    "CD15",
]

COLORS = [
    "#E6194B",
    "#3CB44B",
    "#FFE119",
    "#4363D8",
    "#F58231",
    "#911EB4",
    "#46F0F0",
    "#F032E6",
    "#BCF60C",
    "#FABEBE",
    "#008080",
    "#E6BEFF",
    "#9A6324",
]

CT_MARKER_DICT = {
    "cell_type": {
        "B": ["PAX5"],
        "T": ["CD3"],
        "Myeloid": ["CD11b"],
        "Dendritic": ["CD11c"],
        "Macro": ["CD68"],
        "Stroma": ["CD90"],
        "Lymphatic": ["Podoplanin"],
        "Vascular_CD31": ["CD31"],
        "Vascular_CD34": ["CD34"],
        "NK_CD56": ["CD56"],
        "NK_CD57": ["CD57"],
        "Plasma": ["CD138"],
        "Granulo": ["CD15"],
    }
}


@dataclass
class AstirAnalysisOutput:
    ds: Any
    ds_processed: Any
    ds_with_predictions: Any
    channels: list[str]
    colors: list[str]
    cell_types: list[str]
    label_color_map: dict[str, str]


TITLE = "Cell Type Prediction - Astir"


def run_analysis(
    sample_id: str = "1-1",
    dataset_path: str | None = None,
    threshold_path: str | None = None,
) -> AstirAnalysisOutput:
    if dataset_path is None or threshold_path is None:
        cfg_dataset_path, cfg_threshold_path = get_sample_paths(sample_id)
        dataset_path = dataset_path or cfg_dataset_path
        threshold_path = threshold_path or cfg_threshold_path

    ds = xr.open_zarr(dataset_path)
    print(ds)

    threshold_df = pd.read_csv(threshold_path, sep=",")
    threshold_df["threshold"] = threshold_df["threshold"].fillna(threshold_df["threshold"].median())

    channel_threshold = threshold_df.set_index("channel")["threshold"].to_dict()
    quantiles = [channel_threshold[channel] for channel in CHANNELS]

    ds_processed = (
        ds.pp[CHANNELS]
        .pp.threshold(quantiles)
        .pp.add_quantification()
        .pp.transform_expression_matrix(method="arcsinh")
    )

    seg = ds_processed["_segmentation"].values
    ds_for_astir = ds_processed.pp.drop_layers("_obs").pp.add_segmentation(seg)

    ds_with_predictions = ds_for_astir.tl.astir(CT_MARKER_DICT)

    base_cell_types = list(CT_MARKER_DICT["cell_type"].keys())
    full_cell_types = base_cell_types + ["Other"]
    full_colors = COLORS + ["darkgray"]

    ds_with_predictions = ds_with_predictions.la.set_label_colors(
        full_cell_types,
        full_colors,
    ).pp.add_observations()

    return AstirAnalysisOutput(
        ds=ds,
        ds_processed=ds_processed,
        ds_with_predictions=ds_with_predictions,
        channels=CHANNELS,
        colors=COLORS,
        cell_types=full_cell_types,
        label_color_map=dict(zip(full_cell_types, full_colors)),
    )


def run_analysis_and_report(sample_id: str, all_sample_ids: list[str]):
    result = run_analysis(sample_id=sample_id)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    astir_root = project_root / "outputs" / "astir"

    output_dir = astir_root / "images" / f"html_report_assets-astir-{sample_id}"
    html_path = astir_root / "reports" / f"celltype_prediction_astir_{sample_id}.html"

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    report_selector_options = [
        {
            "sample_id": sid,
            "report_file": f"celltype_prediction_astir_{sid}.html",
        }
        for sid in all_sample_ids
    ]

    generate_html_report(
        ds=result.ds,
        ds_processed=result.ds_processed,
        ds_with_predictions=result.ds_with_predictions,
        channels=result.channels,
        colors=result.colors,
        cell_types=result.cell_types,
        label_color_map=result.label_color_map,
        output_dir=output_dir,
        html_path=html_path,
        title=f"{TITLE} ({sample_id})",
        report_selector_options=report_selector_options,
        current_sample_id=sample_id,
    )

    print(f"[{sample_id}] Saved HTML report: {html_path.resolve()}")
    print(f"[{sample_id}] Saved images in: {output_dir.resolve()}")


def main():
    sample_ids = get_run_sample_ids()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    for sample_id in sample_ids:
        print(f"Running Astir pipeline for sample: {sample_id}")
        run_analysis_and_report(sample_id, sample_ids)
    update_index_html(project_root)


if __name__ == "__main__":
    main()
