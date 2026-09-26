"""
Build portrait-render/types-index.json: the named MaleCNS v1.0 cell types a
lab can order as a type poster (LAB product), with per-hemisphere counts.

    python build_types_index.py            # writes types-index.json
    python build_types_index.py --worker ../../<paypal-worker>/lab-types.js
                                           # also writes the Worker's copy

Contract (frontend fetches it; do not change the shape):
    {"version":1, "generated":"YYYY-MM-DD",
     "types":[{"t":"EPG","n":46,"l":23,"r":23,"m":0}, ...]}
    t = annotation `type`, n = neurons of that type, l/r/m = somaSide L/R/M
    (n can exceed l+r+m: some neurons have no somaSide).

Size budget: the whole file must stay under 400 KB. All 11,751 named types
come to ~471 KB, so the list is cut, deterministically, in this order:
  1. placeholder names are dropped (`*_unclear*`, `*TBD*`): not settled types;
  2. PROTECTED types are always kept (the 153-neuron roster's types, the
     connectome-art subjects, and the 20 LAB sample types);
  3. the rest are ranked: proper names before provisional `CB<digits>`
     names, then more neurons first, then larger share of neurons with
     statusLabel "Reviewed", then name -- and kept from the top until the
     file would reach MAX_BYTES.
Output order is alphabetical by name (case-insensitive) for the frontend.
"""
import os
import re
import sys
import json
import argparse
import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANNOT = os.path.join(HERE, "data", "body-annotations.feather")
ROSTER_JS = os.path.join(HERE, "..", "neurons.js")
OUT = os.path.join(HERE, "types-index.json")
MAX_BYTES = 395_000          # hard contract limit is 400 KB; keep headroom

PLACEHOLDER = re.compile(r"unclear|TBD")
PROVISIONAL = re.compile(r"^CB\d+$")

CONNECTOME_ART = ["EPG", "KCg-m", "T4a", "LC12", "DNp01"]

# The 20 LAB sample types (lab-poster-samples/). One line each on why a
# neuroscientist would ask for it by name.
SAMPLE_TYPES = [
    ("EPG", "head-direction 'compass' neurons of the ellipsoid body (Seelig & Jayaraman 2015); connectome-art subject"),
    ("PEN_a(PEN1)", "ring-attractor shift neurons that rotate the heading bump (Green et al. 2017; Turner-Evans et al. 2017)"),
    ("Delta7", "protocerebral-bridge inhibitory ring neurons of the CX connectome (Hulse et al. 2021)"),
    ("PFL3", "goal-directed steering output of the central complex (Westeinde et al. 2024; Mussells Pires et al. 2024)"),
    ("hDeltaB", "fan-shaped body columnar neurons in the vector-computation circuit (Lyu et al. 2022; Hulse et al. 2021)"),
    ("KCg-m", "gamma-lobe Kenyon cells, the mushroom body memory neurons; connectome-art subject"),
    ("APL", "the single giant GABAergic neuron that gates the whole mushroom body (Liu & Davis 2009)"),
    ("MBON01", "mushroom body output neuron in the Aso et al. 2014 MBON naming used by hemibrain/neuPrint"),
    ("DA1_lPN", "cVA pheromone projection neurons, a sexually dimorphic male circuit (Datta et al. 2008)"),
    ("T4a", "ON-motion direction-selective neurons of the optic lobe (Maisak et al. 2013); connectome-art subject"),
    ("Mi1", "medulla input to the ON-motion pathway, textbook connectome case (Takemura et al. 2013)"),
    ("LPLC2", "looming-detector visual projection neurons driving escape (Klapoetke et al. 2017)"),
    ("LC12", "lobula columnar small-object detectors; connectome-art subject"),
    ("LC10a", "visual projection neurons for male courtship pursuit (Ribeiro et al. 2018)"),
    ("HSE", "horizontal-system lobula plate tangential cell, classic optic-flow neuron"),
    ("DNp01", "the Giant Fiber escape command neuron; connectome-art subject"),
    ("MDN", "moonwalker descending neurons for backward walking (Bidaye et al. 2014)"),
    ("DNa02", "descending steering neurons (Rayshubskiy et al. 2020)"),
    ("pIP10", "descending courtship-song command neuron, male circuit (von Philipsborn et al. 2011)"),
    ("s-LNv", "PDF clock neurons, the 'morning oscillator' (Grima et al. 2004; Stoleru et al. 2004)"),
]


def roster_types(ann):
    """Annotation types of the roster's bodyIds (the roster's own type
    strings are sometimes shortened, e.g. PEN_a vs PEN_a(PEN1))."""
    with open(ROSTER_JS, encoding="utf-8") as fh:
        src = fh.read()
    obj = json.loads(src[src.index("{"): src.rindex("}") + 1])
    ids = {int(n["bodyId"]) for n in obj["neurons"]}
    return set(ann.loc[ann["bodyId"].isin(ids), "type"].astype(str))


def build(max_bytes=MAX_BYTES, today=None):
    ann = pd.read_feather(ANNOT).drop_duplicates("bodyId")
    ann = ann[ann["type"].notna()].copy()
    ann["type"] = ann["type"].astype(str).str.strip()
    ann = ann[ann["type"] != ""]

    rows = []
    for t, g in ann.groupby("type", sort=True):
        side = g["somaSide"]
        rows.append({
            "t": t, "n": int(len(g)),
            "l": int((side == "L").sum()), "r": int((side == "R").sum()),
            "m": int((side == "M").sum()),
            "_rev": float((g["statusLabel"] == "Reviewed").mean()),
        })
    all_named = len(rows)
    rows = [r for r in rows if not PLACEHOLDER.search(r["t"])]
    n_placeholder = all_named - len(rows)

    protected = (roster_types(ann) | set(CONNECTOME_ART)
                 | {t for t, _ in SAMPLE_TYPES})
    protected = {t for t in protected if not PLACEHOLDER.search(t)}
    names = {r["t"] for r in rows}
    missing = sorted(t for t in {t for t, _ in SAMPLE_TYPES} | set(CONNECTOME_ART)
                     if t not in names)
    if missing:
        raise SystemExit("sample/connectome-art types missing from annotations: %s" % missing)

    def rank(r):
        return (0 if r["t"] in protected else 1,
                1 if PROVISIONAL.match(r["t"]) else 0,
                -r["n"], -r["_rev"], r["t"])

    rows.sort(key=rank)
    today = today or datetime.date.today().isoformat()
    head = '{"version":1,"generated":"%s","types":[' % today
    size = len(head) + 2
    kept = []
    for r in rows:
        e = json.dumps({k: r[k] for k in ("t", "n", "l", "r", "m")},
                       separators=(",", ":"), ensure_ascii=False)
        add = len(e.encode("utf-8")) + (1 if kept else 0)
        if size + add > max_bytes:
            if r["t"] in protected:
                raise SystemExit("budget too small for protected types")
            break
        kept.append(r)
        size += add
    dropped = rows[len(kept):]
    kept.sort(key=lambda r: (r["t"].lower(), r["t"]))
    doc = {"version": 1, "generated": today,
           "types": [{k: r[k] for k in ("t", "n", "l", "r", "m")} for r in kept]}
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    stats = {
        "named_types": all_named, "placeholder_dropped": n_placeholder,
        "kept": len(kept), "budget_dropped": len(dropped),
        "budget_dropped_provisional_CB": sum(1 for r in dropped if PROVISIONAL.match(r["t"])),
        "budget_dropped_max_n": max((r["n"] for r in dropped
                                     if not PROVISIONAL.match(r["t"])), default=0),
        "protected": len(protected), "bytes": len(text.encode("utf-8")),
        "neurons_covered": sum(r["n"] for r in kept),
    }
    return text, doc, stats


def worker_module(doc):
    """lab-types.js for the checkout Worker: the same names, newline-joined."""
    names = "\n".join(t["t"] for t in doc["types"])
    assert "`" not in names and "\\" not in names and "${" not in names
    return ("// GENERATED by the-parallel/portrait-render/build_types_index.py from\n"
            "// types-index.json (generated %s, %d types). Do not edit by hand:\n"
            "// regenerate both files together (see README \"LAB product\").\n"
            "export const LAB_TYPES_GENERATED = %s;\n"
            "export const LAB_TYPE_NAMES = new Set(`%s`.split(\"\\n\"));\n"
            % (doc["generated"], len(doc["types"]), json.dumps(doc["generated"]), names))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--worker", default=None, help="also write the Worker's lab-types.js here")
    ap.add_argument("--date", default=None, help="override 'generated' (YYYY-MM-DD)")
    a = ap.parse_args(argv)
    text, doc, stats = build(today=a.date)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    if a.worker:
        with open(a.worker, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(worker_module(doc))
    print(json.dumps(stats))


if __name__ == "__main__":
    sys.exit(main())
