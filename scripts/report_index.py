#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import html
import json
import re

REPORT_PATTERN = re.compile(r"^celltype_prediction_(astir|argmax)_(.+)\.html$")


def collect_reports(project_root: Path) -> dict[str, dict[str, str]]:
    outputs_root = project_root / "outputs"
    report_map: dict[str, dict[str, str]] = {"astir": {}, "argmax": {}}

    for report_type in ("astir", "argmax"):
        report_dir = outputs_root / report_type / "reports"
        if not report_dir.exists():
            continue

        for path in sorted(report_dir.glob("celltype_prediction_*.html")):
            match = REPORT_PATTERN.match(path.name)
            if not match:
                continue
            _, sample_id = match.groups()
            relative_path = path.relative_to(outputs_root).as_posix()
            report_map[report_type][sample_id] = relative_path

    return report_map


def build_index_html(report_map: dict[str, dict[str, str]]) -> str:
    map_json = json.dumps(report_map)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>TNHL Report Launcher</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --card: #111922;
      --line: #253242;
      --text: #e6edf5;
      --muted: #8aa0b8;
      --accent: #58c4dc;
      --accent-2: #a7e35f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(1200px 600px at 15% -10%, #14334a 0%, transparent 55%),
        radial-gradient(900px 500px at 90% 110%, #314114 0%, transparent 60%),
        var(--bg);
      color: var(--text);
      font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;
      padding: 20px;
    }}
    .card {{
      width: min(680px, 100%);
      background: color-mix(in oklab, var(--card) 92%, black 8%);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0.01em;
    }}
    p {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .field {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .field label {{
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    select, button {{
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #0d141c;
      color: var(--text);
      padding: 10px 12px;
      font-size: 14px;
    }}
    button {{
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      color: #05111a;
      border: none;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.12s ease, filter 0.12s ease;
      margin-top: 14px;
    }}
    button:hover {{
      transform: translateY(-1px);
      filter: brightness(1.06);
    }}
    .status {{
      margin-top: 10px;
      color: var(--muted);
      min-height: 20px;
      font-size: 13px;
    }}
    @media (max-width: 700px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class=\"card\">
    <h1>Report Launcher</h1>
    <p>Select core and report type, then open the report in a new tab.</p>

    <div class=\"grid\">
      <div class=\"field\">
        <label for=\"core-select\">Core</label>
        <select id=\"core-select\"></select>
      </div>

      <div class=\"field\">
        <label for=\"report-type-select\">Report Type</label>
        <select id=\"report-type-select\">
          <option value=\"astir\">Astir</option>
          <option value=\"argmax\">Argmax</option>
        </select>
      </div>
    </div>

    <button id=\"open-report\" type=\"button\">Open Report</button>
    <div id=\"status\" class=\"status\"></div>
  </main>

  <script>
    const reportMap = {map_json};

    const coreSelect = document.getElementById("core-select");
    const reportTypeSelect = document.getElementById("report-type-select");
    const openButton = document.getElementById("open-report");
    const statusEl = document.getElementById("status");

    const allCores = Array.from(
      new Set([
        ...Object.keys(reportMap.astir || {{}}),
        ...Object.keys(reportMap.argmax || {{}}),
      ])
    ).sort();

    function renderCoreOptions() {{
      const selectedType = reportTypeSelect.value;
      const available = Object.keys(reportMap[selectedType] || {{}});
      const coresToShow = available.length ? available : allCores;

      const previous = coreSelect.value;
      coreSelect.innerHTML = "";
      coresToShow.forEach((core) => {{
        const opt = document.createElement("option");
        opt.value = core;
        opt.textContent = core;
        coreSelect.appendChild(opt);
      }});

      if (previous && coresToShow.includes(previous)) {{
        coreSelect.value = previous;
      }}
    }}

    function openSelectedReport() {{
      const reportType = reportTypeSelect.value;
      const core = coreSelect.value;
      const target = reportMap[reportType]?.[core];

      if (!target) {{
        statusEl.textContent = `No ${{reportType.toUpperCase()}} report found for core ${{core}}.`;
        return;
      }}

      const url = new URL(target, window.location.href).href;
      window.open(url, "_blank", "noopener,noreferrer");
      statusEl.textContent = `Opened ${{reportType.toUpperCase()}} report for core ${{core}}.`;
    }}

    reportTypeSelect.addEventListener("change", renderCoreOptions);
    openButton.addEventListener("click", openSelectedReport);

    renderCoreOptions();
  </script>
</body>
</html>
"""


def update_index_html(project_root: Path):
    outputs_root = project_root / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    index_path = outputs_root / "index.html"

    report_map = collect_reports(project_root)
    html_text = build_index_html(report_map)
    index_path.write_text(html_text, encoding="utf-8")

    print(f"Updated launcher index: {index_path.resolve()}")


if __name__ == "__main__":
    update_index_html(Path(__file__).resolve().parent.parent)
