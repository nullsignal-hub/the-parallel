"""
"Your neuron" -- deterministic personal portrait from a date.

Demonstrates the storefront mechanic: a buyer types a date (birthday,
anniversary, a child's due date) and gets one specific, real neuron out of
MaleCNS. Same date always returns the same neuron, so the artwork is stable,
re-orderable, and giftable.

The neuron's skeleton is pulled from the public bucket *on demand* -- nothing
is precomputed, which is what makes a web version cheap to run.

The pool is deliberately narrower than "every neuron with a soma":

  * it must have a named cell **type**. An unnamed neuron cannot be given a
    sentence, and a product that sometimes says "TYPE nan" is not a product.
  * its tracing status must not be orphaned, partial or leaves-only.
  * the reconstruction must survive the same outlier test the posters use:
    a merged or mislocated skeleton is rejected and the next candidate in the
    deterministic sequence is taken instead.

**The pool is frozen** in `pool-v1.json` and read from there, never rebuilt at
run time. The date -> neuron mapping depends on the pool's membership *and its
order*, so a rule change silently re-deals every customer's neuron; a reorder
would then arrive as a different picture. See `freeze_pool.py`. The version is
printed on the artwork so any print can be traced back to the pool that made
it. To change the rules, cut a new pool version -- do not touch an old one.

Data: MaleCNS v1.0, CC BY 4.0. See ATTRIBUTION.md.
"""
import os
import hashlib
import argparse
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import render as R
import freeze_pool
from render import (load_swc, project, CREDIT, STYLES, _norm,     # noqa: F401
                    neuron_metrics, outlier_mask, parse_size)

HERE = os.path.dirname(os.path.abspath(__file__))
SWC_DIR = os.path.join(HERE, "data", "swc")
OUT_DIR = os.path.join(HERE, "out")
ANNOT = os.path.join(HERE, "data", "body-annotations.feather")
URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
       "skeletons-malecns/skeletons-swc/{b}.swc")

# Which frozen pool this run sells from. Bump only by publishing a new edition.
POOL_VERSION = "v1"

# A skeleton with fewer segments than this is a stub, not a neuron.
MIN_SEGMENTS = 120

# plain-language gloss so the buyer gets a sentence, not a code
GLOSS = {
    "ol_intrinsic": "a cell of the optic lobe, where the fly turns light into motion",
    "cb_intrinsic": "a central-brain cell, where the fly decides things",
    "vnc_intrinsic": "a nerve-cord cell, wiring the fly's six legs and wings",
    "visual_projection": "a projection cell, carrying vision inward",
    "descending_neuron": "a descending cell, carrying commands down to the body",
    "ascending_neuron": "an ascending cell, carrying the body's news up to the brain",
    "vnc_motor": "a motor cell -- one of the ones that actually moves a muscle",
    "cb_motor": "a motor cell, driving a muscle of the head",
    "cb_sensory": "a sensory cell, the fly's contact with the outside",
    "vnc_sensory": "a sensory cell of the body wall",
    "ol_sensory": "a photoreceptor, where light first becomes a signal",
    "visual_centrifugal": "a centrifugal cell, sending word back out to the eye",
    "cb_endocrine": "an endocrine cell, speaking to the body in hormones",
    "vnc_endocrine": "an endocrine cell of the nerve cord",
    "vnc_efferent": "an efferent cell, carrying orders out of the nerve cord",
    "cb_efferent": "an efferent cell, carrying orders out of the brain",
}

_POOL = {}
_META = {}


def fetch(body_id):
    p = os.path.join(SWC_DIR, "%d.swc" % body_id)
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        os.makedirs(SWC_DIR, exist_ok=True)
        with urllib.request.urlopen(URL.format(b=body_id), timeout=60) as r:
            data = r.read()
        with open(p, "wb") as fh:
            fh.write(data)
    return p


def get_pool(version=None):
    """The frozen pool, in its frozen order, with annotation columns attached.

    The bodyIds and their order come from `pool-<version>.json` and nothing
    else. The annotation file only supplies descriptive columns (type,
    superclass, somaSide); if it were ever revised, the *identity* of each
    date's neuron would still be unchanged, because identity lives in the
    frozen list.
    """
    version = version or POOL_VERSION
    if version not in _POOL:
        ids, meta = freeze_pool.load(version)
        ann = pd.read_feather(ANNOT).drop_duplicates("bodyId").set_index("bodyId")
        missing = np.setdiff1d(ids, ann.index.to_numpy())
        if len(missing):
            raise SystemExit(
                "pool %s names %d bodyIds that are absent from the annotation "
                "file on disk (e.g. %s). The annotations do not match the pool."
                % (version, len(missing), missing[:3]))
        p = ann.reindex(ids).reset_index()          # frozen order preserved
        _POOL[version] = p
        _META[version] = meta
    return _POOL[version]


def get_meta(version=None):
    version = version or POOL_VERSION
    get_pool(version)
    return _META[version]


def pick(date_str, pool=None, attempt=0):
    """Date -> one neuron. Deterministic, uniform, and un-gameable.

    `attempt` walks a deterministic sequence, so a rejected candidate always
    resolves to the *same* replacement for that date.
    """
    pool = get_pool() if pool is None else pool
    h = hashlib.sha256(("%s#%d" % (date_str, attempt)).encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    return pool.iloc[n % len(pool)]


def _family_ids(me, pool, siblings):
    fam = pool[(pool["type"] == me["type"]) & (pool["bodyId"] != me["bodyId"])]
    # same hemisphere only: the mirror-image copy on the other side sits
    # far away and blows the bounding box open, leaving the hero tiny.
    if pd.notna(me.get("somaSide")):
        same = fam[fam["somaSide"] == me["somaSide"]]
        if len(same):
            fam = same
    return fam["bodyId"].astype(int).head(siblings).tolist()


def choose(date_str, siblings=40, max_attempts=12, verbose=False):
    """Pick a neuron and prove it is fit to print before returning it.

    Returns (row, hero_skeleton, ghost_skeletons).
    """
    pool = get_pool()
    rejected = []
    for attempt in range(max_attempts):
        me = pick(date_str, pool, attempt)
        body_id = int(me["bodyId"])
        try:
            hero = load_swc(fetch(body_id))
        except Exception as e:                                 # noqa: BLE001
            rejected.append((body_id, "fetch:%s" % type(e).__name__))
            continue
        if len(hero[0]) < MIN_SEGMENTS:
            rejected.append((body_id, "stub (%d segments)" % len(hero[0])))
            continue

        fam_ids = _family_ids(me, pool, siblings)
        ghosts, gids = [], []
        for b in fam_ids:
            try:
                ghosts.append(load_swc(fetch(b)))
                gids.append(b)
            except Exception:                                  # noqa: BLE001
                continue

        # Judge the hero against its own family: a merged or mislocated
        # reconstruction is exactly what this test was built to catch.
        if len(ghosts) >= 4:
            mask = outlier_mask([hero] + ghosts)
            if mask[0]:
                rejected.append((body_id, "outlier vs its own type"))
                continue
            ghosts = [g for g, bad in zip(ghosts, mask[1:]) if not bad]
        if verbose and rejected:
            for b, why in rejected:
                print("   rejected bodyId %d: %s" % (b, why))
        return me, hero, ghosts

    raise SystemExit("no printable neuron for %s after %d attempts: %s"
                     % (date_str, max_attempts, rejected))


def render_portrait(date_str, style="ink", siblings=40, dpi=150,
                    view="frontal", outfile=None, size=None, verbose=False):
    me, hero, ghosts = choose(date_str, siblings=siblings, verbose=verbose)
    body_id = int(me["bodyId"])
    ctype = str(me["type"])
    sclass = str(me["superclass"])

    size = size or (10, 13)
    R.SCALE = float(np.hypot(*size) / R.REF_DIAGONAL)

    dark = style in ("spectral", "constellation")
    bg, fg, ghost_c, hero_c = (
        ("#07080d", "#e7ecf5", "#2a3550", "#ffd27f") if dark
        else ("#f4f0e6", "#1b1815", "#c9c2b2", "#b3321f"))

    rect = [0.08, 0.11, 0.84, 0.78]
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_axes(rect)
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    for segs, rad in ghosts:                      # the family, receding
        p, _ = project(segs, view)
        ax.add_collection(LineCollection(p, colors=ghost_c,
                                         linewidths=R._lw(0.3), alpha=0.5))
    segs, rad = hero                              # the one that is yours
    p, _ = project(segs, view)
    ax.add_collection(LineCollection(p, colors=hero_c,
                                     linewidths=R._lw(0.4 + 1.6 * _norm(rad)),
                                     alpha=0.95, capstyle="round"))

    # Frame on the hero, not on the whole family: the buyer bought one neuron,
    # so it has to be the thing that fills the paper.
    hp = project(hero[0], view)[0].reshape(-1, 2)
    (x0, y0), (x1, y1) = hp.min(axis=0), hp.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * 0.72          # 44% air around the subject
    box = np.array([[cx - half, cy - half], [cx + half, cy + half]])
    R._fit_limits(ax, box, size, rect, pad=0.0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    gloss = GLOSS.get(sclass, "a cell of the central nervous system")
    fig.text(0.08, 0.945, date_str, color=fg, fontsize=R._fs(30), ha="left")
    fig.text(0.08, 0.918, "YOUR NEURON  ·  BODY ID %d  ·  TYPE %s"
             % (body_id, ctype), color=fg, fontsize=R._fs(8), alpha=0.8,
             ha="left")
    fig.text(0.08, 0.070, gloss.capitalize() + ".",
             color=fg, fontsize=R._fs(9.5), alpha=0.85, ha="left")
    # The pool version belongs on the artwork, not only in a database: it is
    # what lets a reorder years later be proved to be the same neuron.
    fig.text(0.08, 0.048,
             "One of %s named neurons in the male fruit fly.  ·  POOL %s"
             % (f"{len(get_pool()):,}", POOL_VERSION.upper()),
             color=fg, fontsize=R._fs(7), alpha=0.6, ha="left")
    fig.text(0.92, 0.030, CREDIT, color=fg, fontsize=R._fs(4.2), alpha=0.5,
             ha="right")

    os.makedirs(OUT_DIR, exist_ok=True)
    outfile = outfile or os.path.join(
        OUT_DIR, "portrait_%s_%s.png" % (date_str.replace("-", ""), style))
    fig.savefig(outfile, dpi=dpi, facecolor=bg)
    plt.close(fig)
    R.SCALE = 1.0
    return outfile, body_id, ctype, sclass


def main():
    global POOL_VERSION
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="*", default=["1998-04-23"])
    ap.add_argument("--style", default="ink")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--print", dest="print_size", type=parse_size, default=None,
                    help="exact paper size in inches, e.g. 24x36")
    ap.add_argument("--audit", type=int, default=0,
                    help="instead of drawing, sample N dates and report how "
                         "often the pool yields something unprintable")
    ap.add_argument("--pool-version", default=POOL_VERSION,
                    help="which frozen pool edition to sell from")
    a = ap.parse_args()
    POOL_VERSION = a.pool_version

    if a.audit:
        import datetime
        pool = get_pool()
        meta = get_meta()
        print("pool %s: %d neurons, frozen %s, sha256 %s"
              % (meta["pool_version"], len(pool), meta["frozen_on"],
                 meta["sha256"][:16]))
        bad = 0
        for i in range(a.audit):
            d = (datetime.date(1950, 1, 1)
                 + datetime.timedelta(days=(i * 977) % 27000)).isoformat()
            me = pick(d, pool)
            if pd.isna(me["type"]) or str(me["type"]).lower() in ("nan", "none"):
                bad += 1
                print("  UNNAMED for", d)
        print("unnamed types in %d sampled dates: %d" % (a.audit, bad))
        return

    dpi = a.dpi
    if a.print_size and dpi == 150:
        dpi = 300
    for d in a.dates:
        f, b, t, s = render_portrait(d, style=a.style, dpi=dpi,
                                     size=a.print_size, verbose=True)
        print("%s -> bodyId %d  type %s  (%s)\n    %s  %.2f MB"
              % (d, b, t, s, os.path.relpath(f, HERE),
                 os.path.getsize(f) / 1e6))


if __name__ == "__main__":
    main()
