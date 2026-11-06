"""Chart of the retrieval eval (task C15).

    uv run python eval/retrieval_chart.py    # results/retrieval.json → results/retrieval.svg

A plain SVG with no dependencies, so it shows in the README on GitHub. Colors follow the viewer's
light or dark theme. Hovering a bar shows its value, and results/retrieval.md has the full table.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
METRICS = (("recall@10", "Recall@10"), ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10"))
# a method keeps its color in every group
METHODS = (
    ("bm25", "BM25", 1),
    ("dense", "Dense", 2),
    ("hybrid", "Hybrid", 3),
    ("hybrid+rerank", "Hybrid + rerank", 4),
)
HIGHLIGHT = "hybrid+rerank"  # the only method with printed values; the table has the rest

WIDTH, HEIGHT = 640, 340
LEFT, RIGHT, TOP, BOTTOM = 44, 16, 52, 34
BAR = 24
GAP = 2  # surface gap between adjacent bars
STYLE = """
.surface { fill: #fcfcfb; }
.primary { fill: #0b0b0b; }
.secondary { fill: #52514e; }
.grid { stroke: #e4e3df; }
.baseline { stroke: #8a8984; }
.s1 { fill: #2a78d6; } .s2 { fill: #eb6834; } .s3 { fill: #1baf7a; } .s4 { fill: #eda100; }
@media (prefers-color-scheme: dark) {
  .surface { fill: #1a1a19; }
  .primary { fill: #f0efec; }
  .secondary { fill: #c3c2b7; }
  .grid { stroke: #33332f; }
  .baseline { stroke: #6f6e69; }
  .s1 { fill: #3987e5; } .s2 { fill: #d95926; } .s3 { fill: #199e70; } .s4 { fill: #c98500; }
}
text { font: 12px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.title { font-size: 14px; font-weight: 600; }
"""


def _bar(x: float, y: float, height: float, css: str, tip: str) -> str:
    """A column with a 4px rounded top and a square base, with its value as a hover tooltip."""
    r = min(4, height)
    path = (
        f"M{x:g} {y + height:g}V{y + r:g}Q{x:g} {y:g} {x + r:g} {y:g}H{x + BAR - r:g}"
        f"Q{x + BAR:g} {y:g} {x + BAR:g} {y + r:g}V{y + height:g}Z"
    )
    return f'<path class="{css}" d="{path}"><title>{tip}</title></path>'


def chart(results: dict, queries: int) -> str:
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def y_of(value: float) -> float:
        return TOP + plot_h * (1 - value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        'aria-label="Retrieval quality by method and metric">',
        f"<style>{STYLE}</style>",
        f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}"/>',
        f'<text class="primary title" x="{LEFT}" y="20">Retrieval quality by method '
        f"({queries} labeled queries)</text>",
    ]
    x = LEFT
    for _, label, slot in METHODS:
        parts.append(f'<rect class="s{slot}" x="{x}" y="31" width="10" height="10" rx="2"/>')
        parts.append(f'<text class="secondary" x="{x + 15}" y="40">{label}</text>')
        x += 30 + 7 * len(label)
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(tick)
        css = "baseline" if tick == 0 else "grid"
        parts.append(
            f'<line class="{css}" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:g}" y2="{y:g}"/>'
        )
        parts.append(
            f'<text class="secondary" x="{LEFT - 6}" y="{y + 4:g}" '
            f'text-anchor="end">{tick:g}</text>'
        )
    band = plot_w / len(METRICS)
    group = len(METHODS) * BAR + (len(METHODS) - 1) * GAP
    for m, (metric, metric_label) in enumerate(METRICS):
        start = LEFT + m * band + (band - group) / 2
        for i, (name, label, slot) in enumerate(METHODS):
            value = results[name]["mean"][metric]
            bar_x = start + i * (BAR + GAP)
            parts.append(
                _bar(
                    bar_x,
                    y_of(value),
                    plot_h * value,
                    f"s{slot}",
                    f"{label}, {metric_label}: {value:.3f}",
                )
            )
            if name == HIGHLIGHT:
                parts.append(
                    f'<text class="primary" x="{bar_x + BAR / 2:g}" y="{y_of(value) - 5:g}" '
                    f'text-anchor="middle">{value:.2f}</text>'
                )
        parts.append(
            f'<text class="secondary" x="{start + group / 2:g}" y="{HEIGHT - 12}" '
            f'text-anchor="middle">{metric_label}</text>'
        )
    return "\n".join(parts + ["</svg>"]) + "\n"


def main() -> None:
    summary = json.loads((RESULTS / "retrieval.json").read_text(encoding="utf-8"))
    svg = chart(summary["results"], summary["queries"])
    (RESULTS / "retrieval.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
