from __future__ import annotations

from pathlib import Path


def write_markdown_report(path: str | Path, sections: dict[str, dict[str, float]]) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# IBEX 35 Model Report", ""]
    for section, metrics in sections.items():
        lines.append(f"## {section}")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value:.6f}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
