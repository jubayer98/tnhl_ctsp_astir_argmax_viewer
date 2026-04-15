#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

DPI = 600


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def save_fig(fig, path: Path):
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        edgecolor="black",
        facecolor="black",
        transparent=False,
        pil_kwargs={"optimize": True},
    )


def hide_plot_chrome(axis):
    axis.set_axis_off()
    axis.set_facecolor("black")


def orient_prediction_image(image_array: np.ndarray) -> np.ndarray:
    return np.flipud(image_array)


def save_legend(handles, labels, title, path: Path):
    if not handles or not labels:
        return

    legend_h = max(2.5, 0.28 * len(labels) + 1.0)
    fig_leg, ax_leg = plt.subplots(figsize=(4.2, legend_h))
    fig_leg.patch.set_facecolor("black")
    ax_leg.set_facecolor("black")
    ax_leg.axis("off")
    legend = ax_leg.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        title=title,
        fontsize=9,
        title_fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color("white")
    legend.get_title().set_color("white")
    save_fig(fig_leg, path)
    plt.close(fig_leg)


def render_raw_processed(ds, ds_processed, channels, colors, output_dir: Path) -> Path:
    fig_raw_processed, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig_raw_processed.patch.set_facecolor("black")
    fig_raw_processed.patch.set_edgecolor("black")
    _ = ds.pp[channels].pl.colorize(colors).pl.show(ax=ax[0])
    _ = ds_processed.pp[channels].pl.colorize(colors).pl.show(ax=ax[1])

    for axis in ax:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
        hide_plot_chrome(axis)

    ax[0].set_title("Raw", color="white")
    ax[1].set_title("Processed", color="white")

    handles = [Patch(facecolor=color, edgecolor="none", label=channel) for channel, color in zip(channels, colors)]
    fig_raw_processed.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        frameon=False,
        ncol=1,
        fontsize=8,
    )
    for text in fig_raw_processed.legends[0].get_texts():
        text.set_color("white")

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    raw_processed_path = output_dir / "01_raw_processed.png"
    save_fig(fig_raw_processed, raw_processed_path)
    plt.close(fig_raw_processed)

    return raw_processed_path


def render_marker_panels(ds_with_predictions, channels, colors, output_dir: Path):
    channel_colors = dict(zip(channels, colors))
    marker_panels = []

    for marker in channels:
        fig_marker, ax_marker = plt.subplots(figsize=(5.8, 5.8))
        fig_marker.patch.set_facecolor("black")
        _ = ds_with_predictions.pp[[marker]].pl.colorize([channel_colors[marker]]).pl.show(ax=ax_marker)

        legend_marker = ax_marker.get_legend()
        if legend_marker is not None:
            legend_marker.remove()
        hide_plot_chrome(ax_marker)

        fig_marker.tight_layout()
        marker_path = output_dir / f"02_marker_{slugify(marker)}.png"
        save_fig(fig_marker, marker_path)
        plt.close(fig_marker)

        marker_panels.append(
            {
                "marker_name": marker,
                "image_path": marker_path,
            }
        )

    return marker_panels


def build_prediction_views(ds_with_predictions, cell_types, label_color_map, output_dir: Path):
    obs_df = ds_with_predictions.pp.get_layer_as_df(celltypes_to_str=True)
    if "_labels" not in obs_df.columns:
        obs_df = ds_with_predictions["_obs"].to_pandas()

    obs_df = obs_df[["_labels"]].copy()
    obs_df = obs_df[obs_df["_labels"].notna()].copy()
    obs_df["_labels"] = obs_df["_labels"].astype(str)
    obs_df.index = obs_df.index.astype(int)

    segmentation = np.asarray(ds_with_predictions["_segmentation"].values)
    all_cell_ids = obs_df.index.to_numpy(dtype=segmentation.dtype, copy=False)
    all_mask = np.isin(segmentation, all_cell_ids) if all_cell_ids.size else segmentation > 0

    color_map_rgb = {
        label: tuple(int(channel * 255) for channel in to_rgb(color))
        for label, color in label_color_map.items()
    }

    gray_rgb = (110, 110, 110)
    black_rgb = (0, 0, 0)

    existing_labels = set(obs_df["_labels"])
    ordered_prediction_labels = [label for label in cell_types if label in existing_labels]
    present_prediction_labels = ordered_prediction_labels if ordered_prediction_labels else list(dict.fromkeys(obs_df["_labels"]))

    pred_all_path = output_dir / "03_celltype_predictions_all.png"
    pred_legend_path = output_dir / "03_celltype_predictions_legend.png"

    pred_handles = [
        Patch(facecolor=tuple(channel / 255 for channel in color_map_rgb[label]), edgecolor="none")
        for label in present_prediction_labels
        if label in color_map_rgb
    ]

    all_predictions = np.full(segmentation.shape + (3,), black_rgb, dtype=np.uint8)
    for cell_type in present_prediction_labels:
        selected_ids = obs_df.index[obs_df["_labels"] == cell_type].to_numpy(dtype=segmentation.dtype, copy=False)
        if selected_ids.size:
            all_predictions[np.isin(segmentation, selected_ids)] = color_map_rgb.get(cell_type, (31, 41, 55))

    Image.fromarray(orient_prediction_image(all_predictions)).save(pred_all_path, optimize=True)
    save_legend(pred_handles, present_prediction_labels, "Cell Type Legend", pred_legend_path)

    prediction_views = [
        {
            "slug": "all",
            "label": "All Cell Types",
            "image_path": pred_all_path,
        }
    ]

    for cell_type in present_prediction_labels:
        selected_ids = obs_df.index[obs_df["_labels"] == cell_type].to_numpy(dtype=segmentation.dtype, copy=False)
        focused = np.full(segmentation.shape + (3,), black_rgb, dtype=np.uint8)
        focused[all_mask] = gray_rgb
        if selected_ids.size:
            focused[np.isin(segmentation, selected_ids)] = color_map_rgb.get(cell_type, (31, 41, 55))
        focused_path = output_dir / f"03_celltype_{slugify(cell_type)}.png"
        Image.fromarray(orient_prediction_image(focused)).save(focused_path, optimize=True)
        prediction_views.append(
            {
                "slug": slugify(cell_type),
                "label": cell_type,
                "image_path": focused_path,
            }
        )

    return prediction_views, pred_all_path, pred_legend_path


def build_html(
    html_path: Path,
    title: str,
    raw_processed_path: Path,
    marker_panels,
    prediction_views,
    pred_all_path: Path,
    pred_legend_path: Path,
  report_selector_options=None,
  current_sample_id: str | None = None,
):
    prediction_options_html = "\n".join(
        f'                  <option value="{html.escape(view["slug"])}">{html.escape(view["label"])}</option>'
        for view in prediction_views
    )

    prediction_view_map = {
        view["slug"]: {
            "label": view["label"],
            "image": view["image_path"].as_posix(),
        }
        for view in prediction_views
    }
    prediction_view_map_json = json.dumps(prediction_view_map)
    initial_prediction_slug = prediction_views[0]["slug"] if prediction_views else "all"

    selector_items = report_selector_options or []
    show_report_selector = len(selector_items) > 1
    report_selector_map = {
      item["sample_id"]: item["report_file"]
      for item in selector_items
    }
    if current_sample_id is None and selector_items:
      current_sample_id = selector_items[0]["sample_id"]
    report_selector_map_json = json.dumps(report_selector_map)

    report_selector_options_html = "\n".join(
      f'            <option value="{html.escape(item["sample_id"])}"'
      + (" selected" if item["sample_id"] == current_sample_id else "")
      + f'>{html.escape(item["sample_id"])}</option>'
      for item in selector_items
    )

    report_selector_html = ""
    if show_report_selector:
      report_selector_html = f"""
      <div class=\"title-selector\">
      <label for=\"report-sample-select\">CORE</label>
      <select id=\"report-sample-select\">
  {report_selector_options_html}
      </select>
      </div>
  """

    marker_tiles_html = "\n".join(
        f'''          <section class="marker-tile">\n            <header>{html.escape(panel["marker_name"])}<''' +
        f'''/header>\n            <div class="tile-frame sync-frame"><img class="sync-image" src="{html.escape(panel["image_path"].as_posix())}" alt="{html.escape(panel["marker_name"])} marker" /></div>\n          </section>'''
        for panel in marker_panels
    )

    html_text = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #000000;
      --panel: #000000;
      --panel-2: #000000;
      --line: #242424;
      --ink: #edf2f7;
      --muted: #9aa6b2;
      --accent: #7dd3fc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Helvetica, Arial, sans-serif;
      background: #000;
      color: var(--ink);
    }}
    .container {{
      max-width: 1720px;
      margin: 0 auto;
      padding: 12px 14px 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .title-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }}
    .title-row h1 {{
      margin: 0;
    }}
    .title-selector {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .title-selector label {{
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .title-selector select {{
      min-width: 120px;
    }}
    .card {{
      background: #000;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      margin-bottom: 12px;
      box-shadow: none;
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .img-static {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      display: block;
      background: #000;
    }}
    .comparison-layout {{
      display: grid;
      grid-template-columns: minmax(400px, 0.84fr) minmax(0, 1.16fr);
      gap: 12px;
      align-items: start;
    }}
    .marker-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .marker-column {{
      min-width: 0;
    }}
    .marker-tile, .prediction-panel, .side-panel {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
      overflow: hidden;
      min-width: 0;
    }}
    .marker-tile header, .prediction-panel header, .side-panel h3 {{
      padding: 8px 10px;
      margin: 0;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .prediction-stack {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-width: 0;
    }}
    .prediction-panel {{
      display: flex;
      flex-direction: column;
      background: var(--panel-2);
    }}
    .side-panel {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      background: transparent;
      border: 0;
      border-radius: 0;
      overflow: visible;
    }}
    .tile-frame, .prediction-frame {{
      position: relative;
      overflow: hidden;
      background: #000;
      touch-action: none;
      min-height: 0;
    }}
    .tile-frame {{
      flex: 1;
    }}
    .marker-tile .tile-frame {{
      aspect-ratio: 1 / 1;
      min-height: 175px;
    }}
    .prediction-panel .prediction-frame {{
      flex: 0 0 auto;
      height: clamp(390px, 48vh, 560px);
    }}
    .sync-image {{
      position: absolute;
      top: 0;
      left: 0;
      max-width: none;
      max-height: none;
      user-select: none;
      -webkit-user-drag: none;
      transform-origin: 0 0;
    }}
    .panel-controls {{
      display: flex;
      gap: 8px;
      padding: 8px 10px 10px;
      border-top: 1px solid var(--line);
      background: #000;
      flex-wrap: wrap;
      align-items: center;
    }}
    button, select {{
      border: 1px solid #34414d;
      background: #000;
      color: var(--ink);
      border-radius: 10px;
      padding: 7px 9px;
      font-size: 12px;
    }}
    button {{ cursor: pointer; }}
    button:hover, select:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    select {{ width: 100%; }}
    .control-row {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      align-items: start;
    }}
    .selector-card, .legend-box {{
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #000;
      min-width: 0;
    }}
    .selector-card {{
      padding: 10px;
    }}
    .legend-box {{
      margin: 0;
      overflow: hidden;
      background: #000;
      min-height: 0;
      max-width: 220px;
    }}
    .legend-box img {{ display: block; width: 100%; max-width: 220px; background: #000; }}
    @media (max-width: 1380px) {{
      .control-row {{
        grid-template-columns: 1fr;
      }}
      .comparison-layout {{
        grid-template-columns: 1fr;
      }}
      .marker-grid {{
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 1080px) {{
      .marker-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 700px) {{
      .marker-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"title-row\">
      <h1>{html.escape(title)}</h1>
{report_selector_html}
    </div>
    <section class=\"card\">
      <h2>Synchronized View with Processed Markers</h2>
      <div class=\"comparison-layout\">
        <div class=\"prediction-stack\">
          <section class=\"prediction-panel\">
            <header>Cell Type Predictions</header>
            <div class=\"prediction-frame sync-frame\">
              <img id=\"prediction-image\" class=\"sync-image\" src=\"{html.escape(pred_all_path.as_posix())}\" alt=\"Cell Type Predictions\" />
            </div>
            <div class=\"panel-controls\">
              <button id=\"zoom-in\">Zoom In</button>
              <button id=\"zoom-out\">Zoom Out</button>
              <button id=\"reset-view\">Reset View</button>
            </div>
          </section>

          <aside class=\"side-panel\">
            <div class=\"control-row\">
              <section class=\"selector-card\">
                <h3>Cell Type Selector</h3>
                <select id=\"prediction-select\">
{prediction_options_html}
                </select>
              </section>
            </div>
            <div id=\"prediction-legend\" class=\"legend-box\">
              <img src=\"{html.escape(pred_legend_path.as_posix())}\" alt=\"Cell Type Legend\" />
            </div>
          </aside>
        </div>

        <section class=\"marker-column\">
          <div class=\"marker-grid\">
{marker_tiles_html}
          </div>
        </section>
      </div>
    </section>

    <section class=\"card\">
      <h2>Raw Vs Processed</h2>
      <img class=\"img-static\" src=\"{html.escape(raw_processed_path.as_posix())}\" alt=\"Raw vs Processed\" />
    </section>
  </div>

  <script>
    const predictionViews = {prediction_view_map_json};
    const syncFrames = Array.from(document.querySelectorAll('.sync-frame'));
    const frameEntries = syncFrames.map((frame) => ({{ frame, img: frame.querySelector('.sync-image') }}));
    const predictionSelect = document.getElementById('prediction-select');
    const predictionImage = document.getElementById('prediction-image');
    const predictionLegend = document.getElementById('prediction-legend');
    const reportSampleSelect = document.getElementById('report-sample-select');
    const reportSelectorMap = {report_selector_map_json};

    const viewState = {{ zoom: 1, minZoom: 0.6, maxZoom: 24, centerX: 0, centerY: 0 }};
    let dragState = null;

    function getImageDimensions() {{
      const iw = predictionImage.naturalWidth || predictionImage.clientWidth;
      const ih = predictionImage.naturalHeight || predictionImage.clientHeight;
      if (!iw || !ih) return null;
      return {{ iw, ih }};
    }}

    function getBaseFit(frame, dimensions = getImageDimensions()) {{
      if (!dimensions) return null;
      const fw = frame.clientWidth;
      const fh = frame.clientHeight;
      if (!fw || !fh) return null;
      return {{
        frameWidth: fw,
        frameHeight: fh,
        scale: Math.min(fw / dimensions.iw, fh / dimensions.ih),
      }};
    }}

    function applyTransform() {{
      const dimensions = getImageDimensions();
      if (!dimensions) return;
      frameEntries.forEach(({{ frame, img }}) => {{
        const fit = getBaseFit(frame, dimensions);
        if (!fit) return;
        const scale = fit.scale * viewState.zoom;
        const x = fit.frameWidth / 2 - viewState.centerX * scale;
        const y = fit.frameHeight / 2 - viewState.centerY * scale;
        img.style.width = `${{dimensions.iw}}px`;
        img.style.height = `${{dimensions.ih}}px`;
        img.style.transform = `translate(${{x}}px, ${{y}}px) scale(${{scale}})`;
      }});
    }}

    function resetView() {{
      viewState.zoom = 1;
      const dimensions = getImageDimensions();
      if (!dimensions) {{
        viewState.centerX = 0;
        viewState.centerY = 0;
        return;
      }}
      viewState.centerX = dimensions.iw / 2;
      viewState.centerY = dimensions.ih / 2;
      applyTransform();
    }}

    function zoomAround(clientX, clientY, factor, frame) {{
      const dimensions = getImageDimensions();
      const fit = getBaseFit(frame, dimensions);
      if (!fit) return;
      const rect = frame.getBoundingClientRect();
      const localX = clientX - rect.left;
      const localY = clientY - rect.top;
      const scale = fit.scale * viewState.zoom;
      const nextZoom = Math.max(viewState.minZoom, Math.min(viewState.maxZoom, viewState.zoom * factor));
      const nextScale = fit.scale * nextZoom;
      const currentX = fit.frameWidth / 2 - viewState.centerX * scale;
      const currentY = fit.frameHeight / 2 - viewState.centerY * scale;
      const imageX = (localX - currentX) / scale;
      const imageY = (localY - currentY) / scale;
      viewState.centerX = imageX - (localX - fit.frameWidth / 2) / nextScale;
      viewState.centerY = imageY - (localY - fit.frameHeight / 2) / nextScale;
      viewState.zoom = nextZoom;
      applyTransform();
    }}

    function setPredictionView(key) {{
      const view = predictionViews[key] || predictionViews.all;
      const showLegend = key === 'all' || !Object.prototype.hasOwnProperty.call(predictionViews, key);
      predictionSelect.value = Object.prototype.hasOwnProperty.call(predictionViews, key) ? key : 'all';
      predictionImage.onload = () => resetView();
      predictionImage.src = view.image;
      predictionImage.alt = view.label;
      predictionLegend.style.visibility = showLegend ? 'visible' : 'hidden';
      if (showLegend) {{
        predictionLegend.innerHTML = `<img src=\"{html.escape(pred_legend_path.as_posix())}\" alt=\"Cell Type Legend\" />`;
      }}
      if (predictionImage.complete) resetView();
    }}

    syncFrames.forEach((frame) => {{
      frame.addEventListener('wheel', (event) => {{
        event.preventDefault();
        const factor = event.deltaY < 0 ? 1.1 : 0.9;
        zoomAround(event.clientX, event.clientY, factor, frame);
      }}, {{ passive: false }});

      frame.addEventListener('pointerdown', (event) => {{
        dragState = {{
          frame,
          pointerId: event.pointerId,
          lastX: event.clientX,
          lastY: event.clientY,
        }};
        frame.setPointerCapture(event.pointerId);
      }});
      frame.addEventListener('pointermove', (event) => {{
        if (!dragState || dragState.pointerId !== event.pointerId) return;
        const dimensions = getImageDimensions();
        const fit = getBaseFit(frame, dimensions);
        if (!fit) return;
        const scale = fit.scale * viewState.zoom;
        if (!scale) return;
        const dx = event.clientX - dragState.lastX;
        const dy = event.clientY - dragState.lastY;
        viewState.centerX -= dx / scale;
        viewState.centerY -= dy / scale;
        dragState.lastX = event.clientX;
        dragState.lastY = event.clientY;
        applyTransform();
      }});
      frame.addEventListener('pointerup', (event) => {{
        if (dragState && dragState.pointerId === event.pointerId) dragState = null;
      }});
      frame.addEventListener('pointercancel', (event) => {{
        if (dragState && dragState.pointerId === event.pointerId) dragState = null;
      }});
    }});

    document.getElementById('zoom-in').addEventListener('click', () => {{
      const frame = document.querySelector('.prediction-frame');
      const rect = frame.getBoundingClientRect();
      zoomAround(rect.left + rect.width / 2, rect.top + rect.height / 2, 1.15, frame);
    }});
    document.getElementById('zoom-out').addEventListener('click', () => {{
      const frame = document.querySelector('.prediction-frame');
      const rect = frame.getBoundingClientRect();
      zoomAround(rect.left + rect.width / 2, rect.top + rect.height / 2, 0.87, frame);
    }});
    document.getElementById('reset-view').addEventListener('click', resetView);
    predictionSelect.addEventListener('change', (event) => setPredictionView(event.target.value));
    if (reportSampleSelect) {{
      reportSampleSelect.addEventListener('change', (event) => {{
        const selectedSampleId = event.target.value;
        const targetReport = reportSelectorMap[selectedSampleId];
        if (targetReport) {{
          const targetUrl = new URL(targetReport, window.location.href);
          window.location.href = targetUrl.href;
        }}
      }});
    }}
    window.addEventListener('pageshow', () => setPredictionView(predictionSelect.value || '{initial_prediction_slug}'));
    window.addEventListener('resize', resetView);

    if (predictionImage.complete) resetView();
    else predictionImage.addEventListener('load', resetView, {{ once: true }});
    setPredictionView('{initial_prediction_slug}');
  </script>
</body>
</html>
"""

    html_path.write_text(html_text, encoding="utf-8")


def generate_html_report(
    ds,
    ds_processed,
    ds_with_predictions,
    channels,
    colors,
    cell_types,
    label_color_map,
    output_dir: Path,
    html_path: Path,
    title: str,
    report_selector_options=None,
    current_sample_id: str | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_processed_path = render_raw_processed(ds, ds_processed, channels, colors, output_dir)
    marker_panels = render_marker_panels(ds_with_predictions, channels, colors, output_dir)
    prediction_views, pred_all_path, pred_legend_path = build_prediction_views(
        ds_with_predictions,
        cell_types,
        label_color_map,
        output_dir,
    )

    build_html(
        html_path=html_path,
        title=title,
        raw_processed_path=raw_processed_path,
        marker_panels=marker_panels,
        prediction_views=prediction_views,
        pred_all_path=pred_all_path,
        pred_legend_path=pred_legend_path,
        report_selector_options=report_selector_options,
        current_sample_id=current_sample_id,
    )
