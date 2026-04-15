#!/usr/bin/env python

import argparse
from pathlib import Path

from ctsp_astir import run_analysis
from ctsp_html_common import generate_html_report
from report_index import update_index_html

TITLE = "Cell Type Prediction - Astir"


def main():
    parser = argparse.ArgumentParser(description="Generate Astir HTML report for a sample")
    parser.add_argument("--sample", default="1-1", help="Sample ID from ctsp_config.py (example: 1-1)")
    args = parser.parse_args()

    sample_id = args.sample

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    astir_root = project_root / "outputs" / "astir"

    output_dir = astir_root / "images" / f"html_report_assets-astir-{sample_id}"
    html_path = astir_root / "reports" / f"celltype_prediction_astir_{sample_id}.html"

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_analysis(sample_id=sample_id)
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
    )

    print(f"Saved HTML report: {html_path.resolve()}")
    print(f"Saved images in: {output_dir.resolve()}")
    update_index_html(project_root)
    print(f"Updated report launcher: {(project_root / 'index.html').resolve()}")


if __name__ == "__main__":
    main()