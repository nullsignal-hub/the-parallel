"""
LAB -- one MaleCNS v1.0 cell TYPE as a poster (every neuron of the type,
overlaid), for a lab that works on that cell type.

    python type_poster.py --type EPG --style ink --size 18x24 \
        --caption "Jayaraman Lab  ·  SfN 2026" --out out/type.png \
        [--dpi 200] [--max-neurons 250]

Drawing is render.py's type-level artwork, reused as is: render.STYLES,
render._orient (the sheet is turned to match the subject), render._fit_limits,
render.outlier_mask (run per hemisphere, see prune_per_side), render.CREDIT. render.py itself is not modified;
render.load_subject() is not used because it needs data/manifest.csv, which
the repository does not carry -- the type's members come straight from
data/body-annotations.feather here instead.

Deterministic: members are sorted by bodyId; when a type has more than
--max-neurons, an evenly spaced subset of that sorted list is drawn
(np.linspace over the indices). Same inputs -> byte-identical PNG (no
"Software" PNG chunk, no randomness).

Layout, bottom up: the CC BY 4.0 credit on its own line (never omitted,
never overlapped), then "N of M neurons drawn", then the optional caption.

Data: MaleCNS v1.0 (HHMI Janelia FlyEM / Cambridge Connectomics Group /
Google Research), CC BY 4.0.
"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import render as R
from render import load_swc, project, CREDIT, STYLES
import lab_inputs as L
from neuron_sheet import fetch, DEFAULT_SWC_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
ANNOT = os.path.join(HERE, "data", "body-annotations.feather")
MAX_NEURONS = 250
DEFAULT_DPI = 200
VIEW = "frontal"
RECT = [0.07, 0.12, 0.86, 0.78]

SUPERCLASS_WORDS = {
    "ol_intrinsic": "optic lobe intrinsic", "cb_intrinsic": "central brain intrinsic",
    "vnc_intrinsic": "ventral nerve cord intrinsic", "visual_projection": "visual projection",
    "ol_sensory": "optic lobe sensory", "vnc_sensory": "ventral nerve cord sensory",
    "cb_sensory": "central brain sensory", "ascending_neuron": "ascending",
    "descending_neuron": "descending", "vnc_motor": "motor  ·  ventral nerve cord",
    "visual_centrifugal": "visual centrifugal", "sensory_ascending": "sensory ascending",
    "cb_motor": "motor  ·  central brain", "vnc_efferent": "efferent  ·  ventral nerve cord",
    "cb_endocrine": "endocrine  ·  central brain", "vnc_endocrine": "endocrine  ·  ventral nerve cord",
    "sensory_descending": "sensory descending", "efferent_ascending": "efferent ascending",
    "efferent_descending": "efferent descending", "cb_efferent": "efferent  ·  central brain",
}

_ANN = None


def annotations():
    global _ANN
    if _ANN is None:
        a = pd.read_feather(ANNOT, columns=["bodyId", "type", "somaSide", "superclass"])
        a = a[a["type"].notna()].copy()
        a["type"] = a["type"].astype(str).str.strip()
        _ANN = a.drop_duplicates("bodyId")
    return _ANN


def members(cell_type):
    """All bodyIds of the type, ascending, plus the annotation rows."""
    a = annotations()
    rows = a[a["type"] == cell_type].sort_values("bodyId")
    return rows["bodyId"].astype(np.int64).tolist(), rows


def pick(ids, max_neurons):
    """Deterministic, evenly spread subset of an ascending id list."""
    if not max_neurons or len(ids) <= max_neurons:
        return list(ids)
    idx = np.unique(np.linspace(0, len(ids) - 1, max_neurons).round().astype(int))
    return [ids[i] for i in idx]


def subtitle_for(cell_type, rows):
    if cell_type in R.SUBJECT_TITLE:
        return R.SUBJECT_TITLE[cell_type][1]
    sc = rows["superclass"].dropna().astype(str)
    word = SUPERCLASS_WORDS.get(sc.mode().iloc[0], sc.mode().iloc[0].replace("_", " ")) if len(sc) else ""
    side = rows["somaSide"]
    l, r, m = int((side == "L").sum()), int((side == "R").sum()), int((side == "M").sum())
    counts = []
    if l or r:
        counts.append("%d left  ·  %d right" % (l, r))
    if m:
        counts.append("%d midline" % m)
    return "  ·  ".join([p for p in [word] + counts if p])


def load_neurons(ids, swc_dir, workers=8):
    """[(segs, rad)], ids -- SWCs fetched in parallel, assembled in id order."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        paths = list(ex.map(lambda b: fetch(b, swc_dir), ids))
    out, kept = [], []
    for b, p in zip(ids, paths):
        n = load_swc(p)
        if len(n[0]):
            out.append(n)
            kept.append(b)
    return out, kept


def prune_per_side(neurons, ids, rows, verbose=False, what=""):
    """render.drop_outliers, run once per hemisphere (somaSide L / R / other).

    render.outlier_mask's first test ("centroid far from the family's median
    centroid") assumes a family that lives on ONE side. A whole cell type
    spans both hemispheres, and the per-axis median then lands inside the
    larger side's cluster, so the smaller hemisphere is thrown away wholesale
    (measured: KCg-m lost all 117 right-side neurons of 250, DA1_lPN all 6
    right-side ones). Testing each hemisphere as its own family keeps the
    rule's intent (misplaced cells, merges) without that. Order is preserved.
    """
    side = rows.set_index("bodyId")["somaSide"].reindex(ids).astype(object).where(
        lambda s: s.isin(["L", "R"]), "other").to_numpy()
    keep = np.zeros(len(neurons), dtype=bool)
    for s in ("L", "R", "other"):
        idx = np.nonzero(side == s)[0]
        if not len(idx):
            continue
        mask = R.outlier_mask([neurons[i] for i in idx])
        keep[idx[~mask]] = True
        if verbose and mask.any():
            print("   outliers dropped (%s, %s): %d of %d"
                  % (what, s, mask.sum(), len(idx)), file=sys.stderr)
    return ([n for n, k in zip(neurons, keep) if k],
            [b for b, k in zip(ids, keep) if k])


def render_type(cell_type, style, size_key, caption="", dpi=DEFAULT_DPI,
                max_neurons=MAX_NEURONS, outfile=None, swc_dir=DEFAULT_SWC_DIR,
                type_names=None, verbose=False):
    cap, errs = L.validate(cell_type, style, size_key, caption, type_names)
    if errs:
        raise SystemExit("; ".join(errs))
    t0 = time.time()
    ids, rows = members(cell_type)
    if not ids:
        raise SystemExit("no neurons annotated as %s" % cell_type)
    chosen = pick(ids, max_neurons)
    neurons, kept = load_neurons(chosen, swc_dir)
    if not neurons:
        raise SystemExit("no skeletons for %s" % cell_type)
    neurons, kept = prune_per_side(neurons, kept, rows, verbose=verbose, what=cell_type)
    t_load = time.time() - t0

    size = R._orient(L.LAB_SIZES[size_key], neurons, VIEW, verbose=verbose)
    R.SCALE = float(np.hypot(*size) / R.REF_DIAGONAL)
    try:
        fig = plt.figure(figsize=size, dpi=dpi)
        ax = fig.add_axes(RECT)
        bg, fg = STYLES[style](ax, neurons, VIEW)
        fig.patch.set_facecolor(bg)
        ax.set_aspect("equal")
        pts = np.vstack([project(s, VIEW)[0].reshape(-1, 2) for s, _ in neurons])
        R._fit_limits(ax, pts, size, RECT)
        if style != "blueprint":
            ax.set_axis_off()
        else:
            for s in ax.spines.values():
                s.set_color("#1e4a72")
            ax.tick_params(colors="#4e88b8", labelsize=R._fs(5),
                           width=float(R._lw(0.5)), length=3 * R.SCALE)

        fig.text(0.07, 0.950, cell_type, color=fg, fontsize=R._fs(27),
                 fontfamily="DejaVu Sans", fontweight="light", ha="left")
        sub = subtitle_for(cell_type, rows)
        if sub:
            fig.text(0.07, 0.922, sub.upper(), color=fg, fontsize=R._fs(7.5),
                     alpha=0.75, ha="left", fontfamily="DejaVu Sans")
        if cap:
            fig.text(0.07, 0.074, cap, color=fg, fontsize=R._fs(7),
                     alpha=0.85, ha="left", fontfamily="DejaVu Sans")
        drawn = ("%d neurons" % len(neurons) if len(neurons) == len(ids)
                 else "%d of %d neurons drawn" % (len(neurons), len(ids)))
        fig.text(0.07, 0.052, "%s  ·  MaleCNS v1.0  ·  %s view" % (drawn, VIEW),
                 color=fg, fontsize=R._fs(5.5), alpha=0.6, ha="left",
                 fontfamily="DejaVu Sans")
        fig.text(0.93, 0.030, CREDIT, color=fg, fontsize=R._fs(4.3),
                 alpha=0.55, ha="right", fontfamily="DejaVu Sans")

        if outfile is None:
            outfile = os.path.join(R.OUT_DIR, "type_%s_%s_%s.png"
                                   % (cell_type.replace("/", "-"), style, size_key))
        os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
        fig.savefig(outfile, dpi=dpi, facecolor=bg, metadata={"Software": None})
        plt.close(fig)
    finally:
        R.SCALE = 1.0
    return {
        "out": outfile, "type": cell_type, "style": style, "size": size_key,
        "paper_in": "%gx%g" % size, "dpi": dpi, "caption": cap,
        "n_type": len(ids), "n_drawn": len(neurons),
        "n_outliers": len(chosen) - len(neurons),
        "bytes": os.path.getsize(outfile),
        "seconds_load": round(t_load, 2), "seconds_total": round(time.time() - t0, 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--type", dest="cell_type", required=True)
    ap.add_argument("--style", required=True, choices=list(STYLES))
    ap.add_argument("--size", required=True, choices=list(L.LAB_SIZES))
    ap.add_argument("--caption", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--max-neurons", type=int, default=MAX_NEURONS)
    ap.add_argument("--swc-dir", default=DEFAULT_SWC_DIR)
    a = ap.parse_args(argv)
    if not (1 <= a.max_neurons <= 2000):
        raise SystemExit("--max-neurons must be 1..2000")
    if not (30 <= a.dpi <= 300):
        raise SystemExit("--dpi must be 30..300")
    info = render_type(a.cell_type, a.style, a.size, a.caption, dpi=a.dpi,
                       max_neurons=a.max_neurons, outfile=a.out,
                       swc_dir=a.swc_dir, verbose=False)
    print(json.dumps(info))


if __name__ == "__main__":
    sys.exit(main())
