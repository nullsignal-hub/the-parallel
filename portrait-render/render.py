"""
MaleCNS connectome -> print-ready artwork.

Renders real neuron centreline skeletons (SWC) in several *deliberately
different* visual styles. The point of the multiple styles is that the
aesthetic call belongs to a human, not to this script: it produces the
options, it does not rank them.

Data: MaleCNS v1.0 (HHMI Janelia FlyEM / Univ. of Cambridge / MRC LMB /
Google Research), CC BY 4.0. See ATTRIBUTION.md.

Usage:
    python render.py                    # every subject x every style, preview
    python render.py --style ink        # one style
    python render.py --print 24x36 --style spectral --subject EPG
                                        # true print size, 300 dpi
"""
import os
import re
import json
import argparse
import colorsys
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
SWC_DIR = os.path.join(HERE, "data", "swc")
OUT_DIR = os.path.join(HERE, "out")
MANIFEST = os.path.join(HERE, "data", "manifest.csv")
ANNOT = os.path.join(HERE, "data", "body-annotations.feather")
SWC_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
           "skeletons-malecns/skeletons-swc/{b}.swc")

# MaleCNS coordinates are in 8 nm voxels.
NM_PER_UNIT = 8.0


def fetch(body_id):
    """Path to one neuron's SWC, downloading it on first use."""
    p = os.path.join(SWC_DIR, "%d.swc" % int(body_id))
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        os.makedirs(SWC_DIR, exist_ok=True)
        with urllib.request.urlopen(SWC_URL.format(b=int(body_id)),
                                    timeout=60) as r:
            data = r.read()
        with open(p, "wb") as fh:
            fh.write(data)
    return p

# CC BY 4.0 needs three things: credit, a link to the licence, and a statement
# that changes were made. "adapted" carries the third. Wording of the credit
# follows Janelia's own media page.
CREDIT = ("MaleCNS v1.0  ·  data acquired and analyzed by the FlyEM Project Team "
          "at HHMI-Janelia, the Cambridge Connectomics Group and Google Research"
          "  ·  adapted  ·  CC BY 4.0  ·  creativecommons.org/licenses/by/4.0/")

SUBJECT_TITLE = {
    "EPG":   ("EPG", "head-direction compass  ·  ellipsoid body"),
    "KCg-m": ("KCg-m", "Kenyon cells  ·  mushroom body  ·  memory"),
    "T4a":   ("T4a", "motion detectors  ·  optic lobe"),
    "LC12":  ("LC12", "lobula columnar  ·  small-object detection"),
    "DNp01": ("DNp01", "the Giant Fiber  ·  escape command"),
}

# --------------------------------------------------------------------------
# print scaling
# --------------------------------------------------------------------------
# matplotlib line widths and font sizes are in *points* -- i.e. physically
# fixed. Blow the canvas up from a 16" preview to a 43" poster and the same
# 0.4pt hairline becomes 2.6x thinner relative to the drawing, which is
# exactly how a good preview turns into a washed-out print. Everything
# dimensional below is multiplied by SCALE, which is the ratio of the
# canvas diagonal to the reference preview diagonal.
REF_DIAGONAL = np.hypot(10.0, 13.0)          # the preview portrait canvas
SCALE = 1.0                                   # set by render()

# Below this physical width a line stops surviving the print process: ink
# spread on matte paper closes up anything much finer than ~0.25pt. Every
# style clamps to it.
MIN_LW_PT = 0.25


def _lw(v):
    """Scale a line width (points) for the current canvas, with a floor."""
    return np.maximum(np.asarray(v, dtype=float) * SCALE, MIN_LW_PT)


def _fs(v):
    """Scale a font size (points) for the current canvas."""
    return v * SCALE


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def swc_chains(path):
    """Split an SWC tree into unbranched chains of (points, radii).

    A chain runs from a root or a branch point to the next branch point or
    leaf, so drawing them gives continuous strokes instead of thousands of
    disconnected two-point stubs. That matters at print size: disconnected
    stubs with round caps bead up into a string of sausages.
    """
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    nid = arr[:, 0].astype(np.int64)
    xyz = arr[:, 2:5]
    rad = arr[:, 5]
    par = arr[:, 6].astype(np.int64)

    index = {int(n): i for i, n in enumerate(nid)}
    children = {}
    roots = []
    for i, p in enumerate(par):
        j = index.get(int(p), -1)
        if j < 0 or j == i:
            roots.append(i)
        else:
            children.setdefault(j, []).append(i)

    chains = []
    # every chain starts at a root or at a child of a branch point
    starts = list(roots)
    for j, kids in children.items():
        if len(kids) > 1:
            starts.extend(kids)
    seen = set()
    for s in starts:
        if s in seen:
            continue
        chain = [s]
        seen.add(s)
        cur = s
        while True:
            kids = children.get(cur, ())
            if len(kids) != 1 or kids[0] in seen:
                break
            cur = kids[0]
            seen.add(cur)
            chain.append(cur)
        # stitch the chain onto its parent node so strokes actually join up
        head = index.get(int(par[chain[0]]), -1)
        if head >= 0 and head != chain[0]:
            chain = [head] + chain
        if len(chain) >= 2:
            idx = np.array(chain)
            chains.append((xyz[idx], rad[idx]))
    return chains


def _smooth(points, passes=4, lam=0.5):
    """Laplacian smoothing with fixed endpoints.

    SWC nodes sit on the segmentation's voxel lattice, so a raw skeleton is a
    staircase. Invisible in a 150 dpi preview; at 24x36 inches it reads as a
    rendering bug. This straightens the lattice steps without moving the
    neuron: it is a change of *drawing*, not of measurement (see
    ATTRIBUTION.md -- the artworks are declared as adapted, not as figures).
    """
    if len(points) < 3 or passes <= 0:
        return points
    p = points.astype(float).copy()
    for _ in range(passes):
        p[1:-1] += lam * (p[:-2] + p[2:] - 2 * p[1:-1]) * 0.5
    return p


def load_swc(path, smooth=4, max_jump=8.0, jump_p99=4.0):
    """Return (segments, radii) where segments is (N,2,3).

    Long edges are cut out. They are bridges the skeletoniser threw across an
    unsampled gap: the path between the two nodes was never measured, so
    drawing a ruler-straight line there is an interpolation the data does not
    support -- and at print size it is exactly what it looks like, a stray
    straight line through an otherwise organic drawing.

    "Long" has to be judged against the neuron's own sampling, and against its
    own spread, which is why there are two thresholds and the limit is the
    larger of them:

      max_jump  -- multiples of the median node spacing. Right for a densely,
                   evenly sampled skeleton.
      jump_p99  -- multiples of this neuron's own 99th-percentile spacing.
                   This is the one that decides, and it is set so that the
                   rule is a guard rather than a routine filter.

    What the measurements actually say (80 neurons sampled from pool v1, plus
    the five poster subjects):

      * a typical cell is sampled about every 91 units and its longest edge is
        4x that, so the median rule never fires at all;
      * the longest edge in the worst of the 80 is 3.4x that neuron's own p99
        and 3.2 um long -- invisible on paper;
      * the exception is a big, coarsely sampled cell. DNp01, the Giant Fiber,
        runs to 24x its own median, and the median-only rule deleted **14.6%
        of its cable**, breaking the descending axon into dashes on the print.

    So: there are essentially no pathological bridges in this dataset, and the
    rule as first tuned was not removing them, it was shredding large neurons.
    At jump_p99=4 nothing in the sample is cut and DNp01 draws whole; it still
    stands ready for an edge that is genuinely absurd. (An earlier note that
    this rule costs "about 1.5% of drawn length" was measured over the five
    poster subjects and was DNp01 alone -- the other four lose nothing.)

    Set both to 0 to keep every edge.
    """
    chains = swc_chains(path)
    if not chains:
        return np.zeros((0, 2, 3)), np.zeros(0)

    if max_jump or jump_p99:
        d_all = np.concatenate(
            [np.linalg.norm(np.diff(p, axis=0), axis=1) for p, _ in chains])
        limit = max(max_jump * max(float(np.median(d_all)), 1e-9),
                    jump_p99 * max(float(np.percentile(d_all, 99)), 1e-9))
        cut = []
        for pts, rad in chains:
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            brk = np.nonzero(d > limit)[0]
            if not len(brk):
                cut.append((pts, rad))
                continue
            for a, b in zip(np.r_[0, brk + 1], np.r_[brk + 1, len(pts)]):
                if b - a >= 2:
                    cut.append((pts[a:b], rad[a:b]))
        chains = cut

    segs, rads = [], []
    for pts, rad in chains:
        pts = _smooth(pts, passes=smooth)
        rad = _smooth(rad[:, None], passes=smooth)[:, 0] if smooth else rad
        segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
        rads.append((rad[:-1] + rad[1:]) * 0.5)
    if not segs:
        return np.zeros((0, 2, 3)), np.zeros(0)
    return np.concatenate(segs), np.concatenate(rads)


# --------------------------------------------------------------------------
# outlier rejection
# --------------------------------------------------------------------------
# Two things go wrong in an automatically reconstructed connectome, and both
# ruin a print:
#   (a) a neuron annotated as belonging to this side/type actually sits on the
#       far side of the brain. It drags the bounding box open and shrinks the
#       subject to nothing.
#   (b) a reconstruction is a *merge* -- two cells fused by the segmentation --
#       so it has several times the cable and reach of its siblings.
# Both are outliers against the rest of the family, so reject them robustly
# (median/MAD), never against a mean, which the outliers themselves poison.

def neuron_metrics(neurons):
    """(centroid, bbox diagonal, segment count) per neuron."""
    cent, ext, nseg = [], [], []
    for segs, _rad in neurons:
        pts = segs.reshape(-1, 3)
        cent.append(pts.mean(axis=0))
        ext.append(float(np.linalg.norm(np.ptp(pts, axis=0))))
        nseg.append(len(segs))
    return np.array(cent), np.array(ext), np.array(nseg, dtype=float)


def _robust_z(v):
    """|v - median| / (1.4826 * MAD). Zero MAD -> fall back to a ratio rule."""
    v = np.asarray(v, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad <= 1e-9:
        denom = max(abs(med), 1.0) * 0.5      # "more than 50% off the median"
        return np.abs(v - med) / denom
    return np.abs(v - med) / (1.4826 * mad)


def outlier_mask(neurons, z=4.5, min_family=5):
    """True where the neuron should be dropped.

    A family smaller than `min_family` has no usable spread to test against
    (DNp01 is a family of two), so nothing is dropped and the caller keeps
    whatever the data gave it.
    """
    n = len(neurons)
    mask = np.zeros(n, dtype=bool)
    if n < min_family:
        return mask
    cent, ext, nseg = neuron_metrics(neurons)
    # (a) how far this soma/arbour sits from where the family lives
    d = np.linalg.norm(cent - np.median(cent, axis=0), axis=1)
    mask |= _robust_z(d) > z
    # (b) how much bigger/smaller the reconstruction is than its siblings
    mask |= _robust_z(ext) > z
    mask |= _robust_z(nseg) > z
    # never throw the whole family away: if the test would keep less than a
    # third of it, the family is simply heterogeneous and the test is wrong.
    if mask.sum() > n * 2 // 3:
        return np.zeros(n, dtype=bool)
    return mask


def drop_outliers(neurons, z=4.5, ids=None, verbose=False, what=""):
    mask = outlier_mask(neurons, z=z)
    if verbose and mask.any():
        tag = ""
        if ids is not None:
            tag = "  bodyIds " + ", ".join(str(int(ids[i]))
                                           for i in np.nonzero(mask)[0][:8])
        print("   outliers dropped: %d of %d%s%s"
              % (mask.sum(), len(neurons), (" (%s)" % what) if what else "", tag))
    kept = [nrn for nrn, bad in zip(neurons, mask) if not bad]
    if ids is not None:
        return kept, [i for i, bad in zip(ids, mask) if not bad]
    return kept


def load_subject(cell_type, max_neurons=None, prune=True, verbose=False):
    man = pd.read_csv(MANIFEST)
    rows = man[man["cell_type"] == cell_type]
    if max_neurons:
        rows = rows.head(max_neurons)
    out, ids = [], []
    for _, r in rows.iterrows():
        try:
            out.append(load_swc(r["path"]))
            ids.append(r["bodyId"])
        except Exception:                                      # noqa: BLE001
            continue
    if prune and out:
        out, ids = drop_outliers(out, ids=np.array(ids), verbose=verbose,
                                 what=cell_type)
    return out


def project(segs, view="frontal"):
    """3D -> 2D. MaleCNS axes: x lateral, y dorsoventral, z anteroposterior."""
    if view == "frontal":                 # looking at the fly's face
        p = segs[:, :, [0, 1]].copy()
        p[:, :, 1] *= -1
        depth = segs[:, :, 2].mean(axis=1)
    elif view == "dorsal":                # looking down on the head
        p = segs[:, :, [0, 2]].copy()
        depth = segs[:, :, 1].mean(axis=1)
    else:                                 # lateral, from the side
        p = segs[:, :, [2, 1]].copy()
        p[:, :, 1] *= -1
        depth = segs[:, :, 0].mean(axis=1)
    return p, depth


# --------------------------------------------------------------------------
# styles.  Each returns nothing; it draws onto ax.
# Keep them genuinely different in *kind*, not just in hue.
# --------------------------------------------------------------------------
def _norm(v):
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def style_ink(ax, neurons, view):
    """Scientific plate. One weight of line, no colour, paper ground."""
    ax.set_facecolor("#f4f0e6")
    for segs, rad in neurons:
        p, _ = project(segs, view)
        lw = _lw(0.15 + 0.9 * _norm(rad))
        ax.add_collection(LineCollection(p, colors="#1b1815",
                                         linewidths=lw, alpha=0.85,
                                         capstyle="round"))
    return "#f4f0e6", "#1b1815"


def style_spectral(ax, neurons, view):
    """Dark ground, each neuron its own hue, depth sets brightness."""
    ax.set_facecolor("#07080d")
    n = max(len(neurons), 1)
    for i, (segs, rad) in enumerate(neurons):
        p, depth = project(segs, view)
        hue = (0.58 + 0.72 * i / n) % 1.0
        d = _norm(depth)
        cols = np.array([colorsys.hsv_to_rgb(hue, 0.55 + 0.35 * dd,
                                             0.35 + 0.65 * dd) for dd in d])
        cols = np.c_[cols, 0.25 + 0.55 * d]
        lw = _lw(0.2 + 1.1 * _norm(rad))
        ax.add_collection(LineCollection(p, colors=cols, linewidths=lw,
                                         capstyle="round"))
    return "#07080d", "#c9d4e8"


def style_blueprint(ax, neurons, view):
    """Technical drawing: single cyan on deep blue, over a measured grid."""
    bg = "#0a1b2e"
    ax.set_facecolor(bg)
    for segs, rad in neurons:
        p, _ = project(segs, view)
        # faint wide underlay + crisp hairline = engineering-plot feel
        ax.add_collection(LineCollection(p, colors="#2ec5ff",
                                         linewidths=_lw(1.6), alpha=0.10))
        ax.add_collection(LineCollection(p, colors="#7fe3ff",
                                         linewidths=_lw(0.35), alpha=0.9))
    ax.grid(True, color="#1e4a72", linewidth=float(_lw(0.4)), alpha=0.6)
    ax.set_axisbelow(True)
    return bg, "#7fe3ff"


def style_duotone(ax, neurons, view):
    """Risograph-ish: two flat inks, offset registration, no gradients."""
    ax.set_facecolor("#efe9df")
    off = None
    for segs, rad in neurons:
        p, _ = project(segs, view)
        if off is None:
            span = np.ptp(p.reshape(-1, 2), axis=0)
            off = span * 0.006
        ax.add_collection(LineCollection(p + off, colors="#ff4f37",
                                         linewidths=_lw(0.9), alpha=0.55))
        ax.add_collection(LineCollection(p - off, colors="#1c3f94",
                                         linewidths=_lw(0.9), alpha=0.55))
    return "#efe9df", "#1c3f94"


def style_constellation(ax, neurons, view):
    """Abstract: throw away the branches, keep the reach. Gold on black."""
    ax.set_facecolor("#0b0b0c")
    tips, roots = [], []
    for segs, rad in neurons:
        p, _ = project(segs, view)
        pts = p.reshape(-1, 2)
        root = pts[np.argmax(rad)] if len(rad) else pts[0]
        roots.append(root)
        # the extreme points of each neuron = its silhouette
        d = np.linalg.norm(pts - root, axis=1)
        far = pts[np.argsort(d)[-24:]]
        tips.append(far)
        for f in far:
            ax.plot([root[0], f[0]], [root[1], f[1]],
                    color="#d9a441", linewidth=float(_lw(0.22)), alpha=0.30,
                    solid_capstyle="round")
    if tips:
        allt = np.vstack(tips)
        ax.scatter(allt[:, 0], allt[:, 1], s=0.6 * SCALE ** 2, c="#f3d79a",
                   alpha=0.55, linewidths=0)
        allr = np.vstack(roots)
        ax.scatter(allr[:, 0], allr[:, 1], s=14 * SCALE ** 2, c="#ffe9b0",
                   alpha=0.95, linewidths=0, zorder=5)
    return "#0b0b0c", "#d9a441"


STYLES = {
    "ink": style_ink,
    "spectral": style_spectral,
    "blueprint": style_blueprint,
    "duotone": style_duotone,
    "constellation": style_constellation,
}


# --------------------------------------------------------------------------
def _canvas_for(neurons, view):
    """Pick portrait/landscape/square so the neurons actually fill the paper.

    Standard print ratios only -- an artwork that does not match a stock frame
    size is a much worse product than one that does.
    """
    pts = np.vstack([project(s, view)[0].reshape(-1, 2) for s, _ in neurons])
    w, h = np.ptp(pts, axis=0)
    r = w / max(h, 1e-9)
    if r > 1.25:
        return (13, 10)          # landscape  (A-series-ish 1.3:1)
    if r < 0.80:
        return (10, 13)          # portrait
    return (11, 11)              # square


def _orient(size, neurons, view, verbose=False):
    """Turn a requested paper size to match the subject.

    "24x36" names a *sheet*, not an orientation. EPG is a wide subject; drop
    it on a 24-wide, 36-tall sheet and aspect='equal' strands it as a small
    island in the middle of a lot of empty paper. Print shops sell the same
    sheet either way round, so match the paper to the art.
    """
    pts = np.vstack([project(s, view)[0].reshape(-1, 2) for s, _ in neurons])
    w, h = np.ptp(pts, axis=0)
    data_landscape = w / max(h, 1e-9) > 1.0
    paper_landscape = size[0] > size[1]
    if data_landscape != paper_landscape and size[0] != size[1]:
        size = (size[1], size[0])
        if verbose:
            print("   paper turned to %gx%g to match the subject" % size)
    return size


def _fit_limits(ax, pts, size, rect, pad=0.04):
    """Frame the drawing so it fills the paper instead of letterboxing.

    With aspect='equal' the axes would otherwise leave dead bands on two
    sides whenever the data ratio and the paper ratio disagree -- which is
    always, once the paper is a fixed 24x36.
    """
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    dw, dh = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    dw, dh = dw * (1 + 2 * pad), dh * (1 + 2 * pad)
    target = (size[0] * rect[2]) / (size[1] * rect[3])
    if dw / dh < target:
        dw = dh * target
    else:
        dh = dw / target
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.set_xlim(cx - dw / 2, cx + dw / 2)
    ax.set_ylim(cy - dh / 2, cy + dh / 2)


def render(cell_type, style, view="frontal", dpi=150,
           size=None, max_neurons=None, outfile=None, label=None,
           prune=True, verbose=False):
    global SCALE
    neurons = load_subject(cell_type, max_neurons, prune=prune, verbose=verbose)
    if not neurons:
        raise SystemExit("no skeletons for %s -- run fetch_skeletons.py" % cell_type)

    if size is None:
        size = _canvas_for(neurons, view)
    else:
        size = _orient(size, neurons, view, verbose=verbose)
    SCALE = float(np.hypot(*size) / REF_DIAGONAL)

    rect = [0.07, 0.10, 0.86, 0.80]
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_axes(rect)

    bg, fg = STYLES[style](ax, neurons, view)
    fig.patch.set_facecolor(bg)

    ax.set_aspect("equal")
    pts = np.vstack([project(s, view)[0].reshape(-1, 2) for s, _ in neurons])
    _fit_limits(ax, pts, size, rect)
    if style != "blueprint":
        ax.set_axis_off()
    else:
        for s in ax.spines.values():
            s.set_color("#1e4a72")
        ax.tick_params(colors="#4e88b8", labelsize=_fs(5),
                       width=float(_lw(0.5)), length=3 * SCALE)

    title, sub = SUBJECT_TITLE.get(cell_type, (cell_type, ""))
    if label:
        title, sub = label
    fig.text(0.07, 0.955, title, color=fg, fontsize=_fs(27),
             fontfamily="DejaVu Sans", fontweight="light", ha="left")
    fig.text(0.07, 0.928, sub.upper(), color=fg, fontsize=_fs(7.5),
             alpha=0.75, ha="left", fontfamily="DejaVu Sans")
    fig.text(0.07, 0.038,
             "%d neurons  ·  %s view" % (len(neurons), view),
             color=fg, fontsize=_fs(7), alpha=0.7, ha="left")
    fig.text(0.93, 0.038, CREDIT, color=fg, fontsize=_fs(4.3),
             alpha=0.55, ha="right")

    os.makedirs(OUT_DIR, exist_ok=True)
    outfile = outfile or os.path.join(
        OUT_DIR, "%s_%s_%s.png" % (cell_type, style, view))
    fig.savefig(outfile, dpi=dpi, facecolor=bg)
    plt.close(fig)
    SCALE = 1.0
    return outfile


# --------------------------------------------------------------------------
# TWIN PRINTS  (product line S6)
# --------------------------------------------------------------------------
# One brain contains two of nearly everything. For most cell types there is a
# neuron on the left and a neuron on the right that are the *same* cell,
# mirrored: same type, same targets, same job, opposite hemisphere. That is
# Parfit's fission case sitting in the data as a measured fact rather than a
# thought experiment, and it is a two-sheet product: one hemisphere per sheet,
# hung as a pair.
#
# How a pair is established, strongest evidence first:
#
#   tier A -- the annotators named them as counterparts. `instance` carries a
#             side and, where the type is indexed, a position: EPG(PB08)_L5 vs
#             EPG(PB08)_R5, DNp01(GF)_L vs DNp01(GF)_R. Where exactly one cell
#             on each side carries the same name, the pairing is the dataset's,
#             not ours. 5,241 such pairs in pool v1.
#   tier B -- the name is shared by several cells per side (EPG has two L5s).
#             Reflect one side's cell bodies across the midline and match each
#             to its nearest partner. Still within one named instance.
#   tier C -- same cell type only, no instance index. Weakest: these two cells
#             do the same job on opposite sides, but nothing says they are
#             each other's counterpart.
#
# Sell the tier. "The same cell, on both sides of one brain" is true of tier A
# and a guess at tier C, and the difference is the whole value of the object.

# Measured, not assumed: the mean of (x_L + x_R)/2 over the 5,241 tier-A pairs
# in pool v1. Median 48,722 units, s.d. 1,517 (= 12 um), so the two halves of
# this brain really are reflections about a single plane to within a few
# cell-body widths.
MIDLINE_X = 48722.0

# Median |dy| and |dz| between tier-A partners' cell bodies is ~1,600 and
# ~1,300 units (13 and 10 um). A geometric match further off than this is not
# a counterpart, it is the nearest leftover.
PAIR_TOL = 6000.0

_PAIR_KEY = re.compile(r"^(?P<base>.*)_(?P<side>[LRM])(?P<idx>[0-9A-Za-z]*)$")


def _soma_array(series):
    out = np.full((len(series), 3), np.nan)
    for i, v in enumerate(series):
        if v is None:
            continue
        a = np.asarray(v, dtype=float).ravel()
        if a.size == 3:
            out[i] = a
    return out


def load_annotations(pool_version="v1"):
    """Annotation rows restricted to the frozen pool, with soma xyz split out.

    Twin prints are sold out of the same frozen pool as the date portraits, so
    that a customer's neuron and its twin are both guaranteed to exist for as
    long as that pool version is on sale.
    """
    df = pd.read_feather(ANNOT).drop_duplicates("bodyId")
    pool_file = os.path.join(HERE, "pool-%s.json" % pool_version)
    if os.path.exists(pool_file):
        with open(pool_file, encoding="utf-8") as fh:
            ids = json.load(fh)["body_ids"]
        df = df[df["bodyId"].isin(set(ids))]
    df = df.copy()
    xyz = _soma_array(df["somaLocation"])
    df["sx"], df["sy"], df["sz"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    m = df["instance"].astype(str).str.extract(_PAIR_KEY)
    df["pair_base"] = m["base"]
    df["pair_side"] = m["side"]
    df["pair_idx"] = m["idx"].fillna("")
    df["pair_key"] = df["pair_base"].fillna(df["type"].astype(str)) + "|" + \
        df["pair_idx"]
    return df


def find_pairs(cell_type=None, ann=None, pool_version="v1", max_pairs=None):
    """Mirror partners, best-evidence first. One row per pair."""
    if ann is None:
        ann = load_annotations(pool_version)
    if cell_type:
        ann = ann[ann["type"].astype(str) == cell_type]
    ann = ann[ann["pair_side"].isin(["L", "R"]) & ann["sx"].notna()]
    rows = []
    for key, g in ann.groupby("pair_key", sort=True):
        left = g[g["pair_side"] == "L"]
        right = g[g["pair_side"] == "R"]
        if left.empty or right.empty:
            continue
        indexed = bool(g["pair_idx"].iloc[0])
        if len(left) == 1 and len(right) == 1:
            # exactly one cell per side carries this name: the dataset has
            # already said which two are counterparts
            tier = "A"
            pairs = [(left.iloc[0], right.iloc[0])]
        else:
            tier = "B" if indexed else "C"
            pairs = _mirror_match(left, right)
        for l, r in pairs:
            # reflect the right-hand cell body and see how far it lands from
            # the left-hand one: the pair's own measure of how good it is
            err = float(np.linalg.norm(
                np.array([2 * MIDLINE_X - r["sx"], r["sy"], r["sz"]])
                - np.array([l["sx"], l["sy"], l["sz"]])))
            rows.append({
                "type": str(l["type"]),
                "instance": str(l["instance"]),
                "instance_L": str(l["instance"]),
                "instance_R": str(r["instance"]),
                "pair_key": key, "tier": tier,
                "left": int(l["bodyId"]), "right": int(r["bodyId"]),
                "mirror_err_um": err * NM_PER_UNIT / 1000.0,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["tier", "mirror_err_um"]).reset_index(drop=True)
    if max_pairs:
        out = out.head(max_pairs)
    return out


def _mirror_match(left, right):
    """Greedy nearest-partner matching after reflecting the right side."""
    L = left[["sx", "sy", "sz"]].to_numpy(dtype=float)
    R = right[["sx", "sy", "sz"]].to_numpy(dtype=float)
    R = R.copy()
    R[:, 0] = 2 * MIDLINE_X - R[:, 0]
    d = np.linalg.norm(L[:, None, :] - R[None, :, :], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    usedL, usedR, out = set(), set(), []
    for i, j in order:
        if i in usedL or j in usedR or d[i, j] > PAIR_TOL:
            continue
        usedL.add(i)
        usedR.add(j)
        out.append((left.iloc[i], right.iloc[j]))
    return out


def _pair_sheet(neuron, partner, row, side, style, view, size, dpi, rect,
                span, outfile, show_partner):
    """One sheet of a twin pair. Both sheets share `span`, hence one scale."""
    global SCALE
    SCALE = float(np.hypot(*size) / REF_DIAGONAL)
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_axes(rect)

    if show_partner and partner is not None:
        # the other one, reflected onto this sheet: what this cell would look
        # like if it had been born on the other side. Drawn faint, behind.
        segs, rad = partner
        mirrored = segs.copy()
        mirrored[:, :, 0] = 2 * MIDLINE_X - mirrored[:, :, 0]
        p, _ = project(mirrored, view)
        ax.add_collection(LineCollection(
            p, colors="#8a8a8a", linewidths=_lw(0.25), alpha=0.28))

    bg, fg = STYLES[style](ax, [neuron], view)
    fig.patch.set_facecolor(bg)
    ax.set_aspect("equal")

    pts = project(neuron[0], view)[0].reshape(-1, 2)
    cx, cy = (pts.min(axis=0) + pts.max(axis=0)) / 2
    hw, hh = span[0] / 2, span[1] / 2
    box = np.array([[cx - hw, cy - hh], [cx + hw, cy + hh]])
    _fit_limits(ax, box, size, rect, pad=0.06)
    if style != "blueprint":
        ax.set_axis_off()
    else:
        for s in ax.spines.values():
            s.set_color("#1e4a72")
        ax.tick_params(colors="#4e88b8", labelsize=_fs(5),
                       width=float(_lw(0.5)), length=3 * SCALE)

    hemi = "LEFT" if side == "L" else "RIGHT"
    body = row["left"] if side == "L" else row["right"]
    twin = row["right"] if side == "L" else row["left"]
    inst = row["instance_%s" % side]
    fig.text(0.07, 0.955, row["type"], color=fg, fontsize=_fs(27),
             fontfamily="DejaVu Sans", fontweight="light", ha="left")
    fig.text(0.07, 0.928, "%s HEMISPHERE  ·  %s" % (hemi, inst),
             color=fg, fontsize=_fs(7.5), alpha=0.75, ha="left")
    # Three stacked lines rather than one: a single footer line ran straight
    # through the right-aligned credit, which is 150 characters long.
    fig.text(0.07, 0.070, "BODY ID %d" % body,
             color=fg, fontsize=_fs(7), alpha=0.7, ha="left")
    fig.text(0.07, 0.052, "ONE OF A PAIR  ·  ITS COUNTERPART IS %d" % twin,
             color=fg, fontsize=_fs(6), alpha=0.5, ha="left")
    fig.text(0.93, 0.030, CREDIT, color=fg, fontsize=_fs(4.3), alpha=0.55,
             ha="right")

    fig.savefig(outfile, dpi=dpi, facecolor=bg)
    plt.close(fig)
    SCALE = 1.0
    return outfile


def render_pair(row, style="ink", view="frontal", dpi=150, size=None,
                show_partner=False, suffix="", verbose=False):
    """Draw both sheets of one twin pair.

    The two sheets are deliberately *not* framed independently. Each is
    centred on its own neuron, but the window is the same size on both, so a
    millimetre on one sheet is a millimetre on the other. Frame them
    separately and the smaller cell is silently enlarged to fill its paper,
    which turns a measured pair into two unrelated pictures -- and the one
    real difference between the twins, that they are not quite the same size,
    is exactly what gets thrown away.
    """
    a = load_swc(fetch(row["left"]))
    b = load_swc(fetch(row["right"]))
    if not len(a[0]) or not len(b[0]):
        raise SystemExit("empty skeleton for pair %s" % row["instance"])

    # Shared window, sized per axis rather than as a square: a square window
    # around a long thin descending neuron is mostly empty paper.
    spans = np.array([np.ptp(project(n[0], view)[0].reshape(-1, 2), axis=0)
                      for n in (a, b)])
    span = spans.max(axis=0)

    if size is None:
        size = _canvas_for([a], view)
    else:
        size = _orient(size, [a], view, verbose=verbose)

    os.makedirs(OUT_DIR, exist_ok=True)
    rect = [0.07, 0.10, 0.86, 0.80]
    outs = []
    for side, me, other in (("L", a, b), ("R", b, a)):
        f = os.path.join(OUT_DIR, "pair_%s_%s_%s_%d%s.png"
                         % (row["type"].replace("/", "-"), style, side,
                            row["left"] if side == "L" else row["right"],
                            suffix))
        outs.append(_pair_sheet(me, other, row, side, style, view, size, dpi,
                                rect, span, f, show_partner))
    if verbose:
        rel = 100.0 * abs(spans[0] - spans[1]).max() / max(span.max(), 1e-9)
        print("   pair %s  tier %s  mirror error %.1f um  "
              "size difference between the twins %.1f%%  shared scale"
              % (row["instance"], row["tier"], row["mirror_err_um"], rel))
    return outs


def parse_size(s):
    """'24x36' -> (24.0, 36.0) inches."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[xX*]\s*(\d+(?:\.\d+)?)\s*", s)
    if not m:
        raise argparse.ArgumentTypeError("size must look like 24x36 (inches)")
    return (float(m.group(1)), float(m.group(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default=None)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--view", default="frontal")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--print", dest="print_size", type=parse_size, default=None,
                    help="exact paper size in inches, e.g. 24x36. Implies 300 dpi "
                         "unless --dpi says otherwise.")
    ap.add_argument("--suffix", default="", help="appended to the output name")
    ap.add_argument("--no-prune", action="store_true",
                    help="keep outlier neurons (for comparison)")
    ap.add_argument("--pair", metavar="TYPE",
                    help="draw a twin pair (two sheets) for this cell type")
    ap.add_argument("--pair-index", type=int, default=0,
                    help="which pair of that type, best-evidence first")
    ap.add_argument("--list-pairs", metavar="TYPE", nargs="?", const="",
                    help="list available twin pairs and stop")
    ap.add_argument("--partner-ghost", action="store_true",
                    help="also draw the counterpart, mirrored and faint")
    a = ap.parse_args()

    dpi = a.dpi
    if a.print_size and dpi == 150:
        dpi = 300

    if a.list_pairs is not None:
        pairs = find_pairs(a.list_pairs or None)
        if not len(pairs):
            raise SystemExit("no pairs found")
        counts = pairs["tier"].value_counts().to_dict()
        print("%d pairs   by tier: %s" % (len(pairs), counts))
        with pd.option_context("display.width", 140,
                               "display.max_rows", 40):
            print(pairs.head(40).to_string(index=True))
        return

    if a.pair:
        pairs = find_pairs(a.pair)
        if not len(pairs):
            raise SystemExit("no L/R pair for %s" % a.pair)
        row = pairs.iloc[a.pair_index]
        styles = [a.style] if a.style else list(STYLES)
        for st in styles:
            outs = render_pair(row, style=st, view=a.view, dpi=dpi,
                               size=a.print_size,
                               show_partner=a.partner_ghost,
                               suffix=a.suffix, verbose=True)
            for p in outs:
                print("wrote", os.path.relpath(p, HERE),
                      "%.2f MB" % (os.path.getsize(p) / 1e6))
        return

    styles = [a.style] if a.style else list(STYLES)
    subjects = [a.subject] if a.subject else list(SUBJECT_TITLE)

    for s in subjects:
        for st in styles:
            out = None
            if a.suffix:
                out = os.path.join(OUT_DIR, "%s_%s_%s%s.png"
                                   % (s, st, a.view, a.suffix))
            p = render(s, st, view=a.view, dpi=dpi, size=a.print_size,
                       outfile=out, prune=not a.no_prune, verbose=True)
            print("wrote", os.path.relpath(p, HERE),
                  "%.2f MB" % (os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    main()
