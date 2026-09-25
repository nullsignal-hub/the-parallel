"""
THE PARALLEL -- one visitor's neuron, drawn as a square print sheet.

The quiz / motion test on the site hands each visitor one of the 153 neurons
in the roster (`../neurons.js`). This draws that single neuron in one of the
five `render.py` styles and lays it out as a square sheet, for printing on
demand after purchase (see `.github/workflows/render-neuron.yml`).

    python neuron_sheet.py --body-id 10001 --style spectral \
        --size 12x12 --dpi 300 --out out/neuron.png

Layout follows `render._pair_sheet`: style-coloured ground, the neuron
framed with a margin, cell type + a short subtitle top left, the bodyId
bottom left, and the CC BY 4.0 credit (`render.CREDIT`, never omitted) along
the bottom.

Deterministic: same bodyId + style + size + dpi -> the same image. Nothing
here is random; the SWC is the published MaleCNS v1.0 skeleton.

Data: MaleCNS v1.0 (HHMI Janelia FlyEM / Cambridge Connectomics Group /
Google Research), CC BY 4.0.
"""
import os
import re
import sys
import json
import time
import argparse
import urllib.request

import colorsys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import render as R
from render import load_swc, project, CREDIT, STYLES, parse_size

HERE = os.path.dirname(os.path.abspath(__file__))
ROSTER_JS = os.path.join(HERE, "..", "neurons.js")
DEFAULT_SWC_DIR = os.environ.get("NEURON_SWC_DIR") or os.path.join(HERE, "data", "swc")

# Axes rectangle on the sheet (figure fractions). Leaves a band on top for
# the title and one at the bottom for bodyId + credit.
RECT = [0.08, 0.13, 0.84, 0.74]
PAD = 0.06           # air around the neuron inside RECT, per side

SIDE_WORD = {"L": "left hemisphere", "R": "right hemisphere", "M": "midline"}


def load_roster(path=ROSTER_JS):
    """{bodyId(int): roster entry} from the site's own neurons.js."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    obj = json.loads(src[src.index("{"): src.rindex("}") + 1])
    return {int(n["bodyId"]): n for n in obj["neurons"]}


def fetch(body_id, swc_dir=DEFAULT_SWC_DIR, tries=3):
    """Path to one neuron's SWC, downloading (with retries) on first use."""
    p = os.path.join(swc_dir, "%d.swc" % int(body_id))
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    os.makedirs(swc_dir, exist_ok=True)
    last = None
    for i in range(tries):
        if i:
            time.sleep(2 * i)
        try:
            with urllib.request.urlopen(R.SWC_URL.format(b=int(body_id)),
                                        timeout=60) as r:
                data = r.read()
            if not data:
                raise IOError("empty SWC")
            tmp = p + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, p)
            return p
        except Exception as e:                                 # noqa: BLE001
            last = e
    raise SystemExit("could not fetch SWC for bodyId %d: %s" % (int(body_id), last))


def subtitle_for(entry):
    if not entry:
        return ""
    parts = [str(entry.get("kind") or "").strip()]
    side = SIDE_WORD.get(entry.get("side"))
    if side:
        parts.append(side)
    return "  ·  ".join(p for p in parts if p)


# --------------------------------------------------------------------------
# Single-neuron versions of two styles.  render.py's spectral and
# constellation were written for a whole cell TYPE (dozens of neurons); on one
# neuron spectral sinks into the dark and constellation collapses to a few
# rays.  These keep each style's ground, colours and spirit but are built for
# one neuron.  render.py itself (type-level art, pair sheets) is untouched.
# The browser preview (parallel-styles.js) mirrors these two functions.
# --------------------------------------------------------------------------
def _root_point(p, rad):
    """Segment start of the thickest segment: the cell's root (soma side)."""
    pts = p.reshape(-1, 2)
    return pts[int(np.argmax(np.repeat(rad, 2)))]


def _degree_keys(p):
    """Unique endpoints (rounded to 0.01) and how many segments meet there."""
    pts = np.round(p.reshape(-1, 2), 2)
    return np.unique(pts, axis=0, return_counts=True)


def style_spectral_single(ax, neurons, view):
    """Dark ground; hue runs from cool at the root to warm at the far tips,
    every line at full brightness, over a soft glow."""
    ax.set_facecolor("#07080d")
    for segs, rad in neurons:
        p, _ = project(segs, view)
        root = _root_point(p, rad)
        dist = np.linalg.norm(p.mean(axis=1) - root, axis=1)
        t = R._norm(dist)
        cols = np.array([colorsys.hsv_to_rgb(0.60 - 0.55 * tt, 0.70, 1.0) for tt in t])
        lw = R._lw(0.2 + 1.1 * R._norm(rad))
        ax.add_collection(LineCollection(p, colors=np.c_[cols, np.full(len(t), 0.12)],
                                         linewidths=lw * 3.2, capstyle="round"))
        ax.add_collection(LineCollection(p, colors=np.c_[cols, np.full(len(t), 0.92)],
                                         linewidths=lw, capstyle="round"))
    return "#07080d", "#c9d4e8"


def style_constellation_single(ax, neurons, view):
    """Gold on black star field: every skeleton point a star sized by its
    thickness, the branch tips brighter, the root a white star."""
    ax.set_facecolor("#0b0b0c")
    for segs, rad in neurons:
        p, _ = project(segs, view)
        pts = p.reshape(-1, 2)
        s = (0.5 + 5.5 * R._norm(np.repeat(rad, 2))) * R.SCALE ** 2
        ax.scatter(pts[:, 0], pts[:, 1], s=s, c="#f3d79a", alpha=0.55, linewidths=0)
        keys, cnt = _degree_keys(p)
        tips = keys[cnt == 1]
        ax.scatter(tips[:, 0], tips[:, 1], s=6 * R.SCALE ** 2, c="#ffe9b0",
                   alpha=0.95, linewidths=0, zorder=4)
        root = _root_point(p, rad)
        ax.scatter([root[0]], [root[1]], s=30 * R.SCALE ** 2, c="#ffffff",
                   alpha=1, linewidths=0, zorder=6)
    return "#0b0b0c", "#d9a441"


SHEET_STYLES = dict(STYLES)
SHEET_STYLES["spectral"] = style_spectral_single
SHEET_STYLES["constellation"] = style_constellation_single


def render_sheet(body_id, style, size=(12.0, 12.0), dpi=300, outfile=None,
                 cell_type=None, subtitle=None, view="frontal",
                 swc_dir=DEFAULT_SWC_DIR):
    if style not in SHEET_STYLES:
        raise SystemExit("unknown style %r (one of %s)" % (style, ", ".join(SHEET_STYLES)))
    neuron = load_swc(fetch(body_id, swc_dir))
    if not len(neuron[0]):
        raise SystemExit("empty skeleton for bodyId %d" % body_id)

    R.SCALE = float(np.hypot(*size) / R.REF_DIAGONAL)
    try:
        fig = plt.figure(figsize=size, dpi=dpi)
        ax = fig.add_axes(RECT)
        bg, fg = SHEET_STYLES[style](ax, [neuron], view)
        fig.patch.set_facecolor(bg)
        ax.set_aspect("equal")

        # Frame on the neuron's own extent (every drawn point, so no branch
        # can fall outside), letterboxed to the axes' shape.
        pts = project(neuron[0], view)[0].reshape(-1, 2)
        R._fit_limits(ax, pts, size, RECT, pad=PAD)
        if style != "blueprint":
            ax.set_axis_off()
        else:
            for s in ax.spines.values():
                s.set_color("#1e4a72")
            ax.tick_params(colors="#4e88b8", labelsize=R._fs(5),
                           width=float(R._lw(0.5)), length=3 * R.SCALE)

        fig.text(0.08, 0.935, cell_type or "", color=fg, fontsize=R._fs(27),
                 fontfamily="DejaVu Sans", fontweight="light", ha="left")
        if subtitle:
            fig.text(0.08, 0.906, subtitle.upper(), color=fg,
                     fontsize=R._fs(7.5), alpha=0.75, ha="left",
                     fontfamily="DejaVu Sans")
        fig.text(0.08, 0.068, "bodyId %d" % body_id, color=fg,
                 fontsize=R._fs(7), alpha=0.7, ha="left",
                 fontfamily="DejaVu Sans")
        fig.text(0.08, 0.050, "MaleCNS v1.0  ·  %s view" % view, color=fg,
                 fontsize=R._fs(5.5), alpha=0.5, ha="left",
                 fontfamily="DejaVu Sans")
        fig.text(0.92, 0.030, CREDIT, color=fg, fontsize=R._fs(4.3),
                 alpha=0.55, ha="right", fontfamily="DejaVu Sans")

        if outfile is None:
            outfile = os.path.join(R.OUT_DIR, "neuron_%d_%s.png" % (body_id, style))
        d = os.path.dirname(os.path.abspath(outfile))
        os.makedirs(d, exist_ok=True)
        # No "Software" chunk: keeps the bytes independent of the matplotlib
        # patch version, so the same input really is the same file.
        fig.savefig(outfile, dpi=dpi, facecolor=bg, metadata={"Software": None})
        plt.close(fig)
    finally:
        R.SCALE = 1.0
    return outfile


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--body-id", required=True)
    ap.add_argument("--style", required=True, choices=list(STYLES))
    ap.add_argument("--size", type=parse_size, default=(12.0, 12.0),
                    help="paper size in inches, e.g. 12x12 (default)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--type", dest="cell_type", default=None,
                    help="override the cell type printed on the sheet "
                         "(default: the roster's type for this bodyId)")
    ap.add_argument("--swc-dir", default=DEFAULT_SWC_DIR)
    ap.add_argument("--allow-off-roster", action="store_true",
                    help="draw a bodyId that is not in neurons.js")
    a = ap.parse_args(argv)

    if not re.fullmatch(r"[0-9]{1,12}", a.body_id):
        raise SystemExit("--body-id must be digits only")
    body_id = int(a.body_id)
    roster = load_roster()
    entry = roster.get(body_id)
    if entry is None and not a.allow_off_roster:
        raise SystemExit("bodyId %d is not in the 153-neuron roster" % body_id)

    cell_type = a.cell_type or (entry and str(entry["type"])) or ""
    out = render_sheet(body_id, a.style, size=a.size, dpi=a.dpi,
                       outfile=a.out, cell_type=cell_type,
                       subtitle=subtitle_for(entry), swc_dir=a.swc_dir)
    print(json.dumps({"out": out, "bodyId": body_id, "type": cell_type,
                      "style": a.style, "bytes": os.path.getsize(out)}))


if __name__ == "__main__":
    main()
