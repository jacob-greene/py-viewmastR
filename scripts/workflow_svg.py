"""Write docs/img/workflow.svg, the cartoon flowchart of the workflow in the README.

    python scripts/workflow_svg.py
"""
from pathlib import Path

import numpy as np

W, H = 1000, 330
out = []
A = out.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  'font-family="Helvetica, Arial, sans-serif">')
A(f'<rect width="{W}" height="{H}" rx="16" fill="#ffffff"/>')
A('<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
  'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>')
PAL = ["#e377c2", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

CENTRES = [(-34, -16), (30, -20), (-4, 22), (38, 20), (-38, 26)]
JIT = {seed: np.random.default_rng(seed).normal(0, 6, (len(CENTRES), 6, 2)) for seed in (1, 2)}


def cells(cx, cy, colored, seed=1, r=6, sx=1.0):
    for k, (dx, dy) in enumerate(CENTRES):
        for jx, jy in JIT[seed][k]:
            x, y = cx + sx * dx + jx, cy + dy + jy
            col = PAL[k] if colored else "#b9b9b9"
            A(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>')


def card(x, y, w, h, title, sub, fill):
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="#d0d0d0"/>')
    A(f'<text x="{x + w / 2}" y="{y + 28}" text-anchor="middle" font-size="19" font-weight="bold" fill="#222">{title}</text>')
    for i, s in enumerate(sub):
        A(f'<text x="{x + w / 2}" y="{y + h - 14 - 18 * (len(sub) - 1 - i)}" text-anchor="middle" font-size="14" fill="#444">{s}</text>')

def arrow(x1, y1, x2, y2):
    A(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#555" stroke-width="3" marker-end="url(#arr)"/>')

# 1 inputs
card(20, 15, 190, 145, "Reference", ["labelled cells"], "#f7f3fb")
cells(115, 78, True, seed=1, r=5)
card(20, 170, 190, 145, "Query", ["unlabelled cells"], "#f4f4f4")
cells(115, 233, False, seed=2, r=5)
# 2 shared genes (Venn)
card(260, 15, 200, 300, "1. Shared genes", ["calculate_feature_dispersion()", "select_features()"], "#fdf6ec")
A('<circle cx="335" cy="160" r="52" fill="#9467bd" fill-opacity="0.35" stroke="#9467bd" stroke-width="2"/>')
A('<circle cx="385" cy="160" r="52" fill="#7f7f7f" fill-opacity="0.30" stroke="#7f7f7f" stroke-width="2"/>')
A('<text x="360" y="166" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">genes</text>')
A('<text x="312" y="226" text-anchor="middle" font-size="12" fill="#555">reference</text>')
A('<text x="410" y="226" text-anchor="middle" font-size="12" fill="#555">query</text>')
arrow(212, 88, 256, 130)
arrow(212, 242, 256, 195)
# 3 train
card(510, 15, 200, 300, "2. Train", ["viewmastR()", "optional first: augment_data()"], "#eef5fb")
ins = [(560, y) for y in (100, 140, 180, 220)]
outs = [(660, y) for y in (90, 125, 160, 195, 230)]
for a in ins:
    for b in outs:
        A(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#9ab" stroke-width="1"/>')
for a in ins:
    A(f'<circle cx="{a[0]}" cy="{a[1]}" r="9" fill="#7f7f7f"/>')
for k, b in enumerate(outs):
    A(f'<circle cx="{b[0]}" cy="{b[1]}" r="9" fill="{PAL[k]}"/>')
A('<text x="610" y="262" text-anchor="middle" font-size="13" fill="#444">genes → cell types</text>')
arrow(462, 165, 506, 165)
# 4 predict
card(760, 15, 220, 300, "3. Label the query", ["query.obs['viewmastR_pred']", "viewmastR_infer() for new data"], "#eef8ef")
cells(870, 150, True, seed=2, r=8, sx=1.4)
arrow(712, 165, 756, 165)
A('</svg>')
open(Path(__file__).resolve().parents[1] / "docs" / "img" / "workflow.svg", "w").write("\n".join(out) + "\n")
