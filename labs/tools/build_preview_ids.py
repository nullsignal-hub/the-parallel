"""Preview bodyIds for the LAB order page (labs/index*.html).

types-index.json only carries counts, but the page has to fetch real skeletons
to draw its preview, so it needs a few bodyIds per cell type. This writes one
small file per type:

    labs/preview/<enc(type)>.json
        {"t": "EPG", "n": 46, "s": "<subtitle>", "ids": [10539, ...], "sd": "LLRR..."}
    s  = the sheet's subtitle line, from portrait-render/type_poster.subtitle_for
         (the print renderer's own function, so preview and print agree)
    sd = somaSide per id, "L" / "R" / "o" (other), for type_poster.prune_per_side

enc(type): every character outside [A-Za-z0-9_-] becomes "~XX" (upper-case hex
of the byte), so "PEN_a(PEN1)" -> "PEN_a~28PEN1~29". The page computes the same
name (labs/index*.html, previewFile()). No two MaleCNS type names differ only by
case, so the files do not collide on case-insensitive disks (checked below).

Which bodyIds (deterministic, so the preview never changes between visits):
  type_poster.members(T) (annotation rows, bodyId ascending) passed through
  type_poster.pick(ids, 30) -- the same evenly spaced rule the print uses
  with 250, so the preview is drawn the same way from a smaller subset.
This is a *sample* for the preview only. The print is rendered server-side
from its own list (up to 250 neurons); the page says so.

Usage (from the repo root):
    python labs/tools/build_preview_ids.py                 # types from labs/types-index.json
    python labs/tools/build_preview_ids.py --index portrait-render/types-index.json
Re-run whenever types-index.json is regenerated (copy it from
portrait-render/types-index.json to labs/types-index.json first).
"""
import os
import re
import json
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
LABS = os.path.dirname(HERE)
REPO = os.path.dirname(LABS)
PR = os.path.join(REPO, "portrait-render")
sys.path.insert(0, PR)
MAX_IDS = 30


def enc(t):
    return "".join(c if re.match(r"[A-Za-z0-9_-]", c) else
                   "".join("~%02X" % b for b in c.encode("utf-8")) for c in t)


def main():
    import type_poster as TP                      # the print renderer
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(LABS, "types-index.json"))
    ap.add_argument("--out", default=os.path.join(LABS, "preview"))
    a = ap.parse_args()

    ann = TP.annotations()
    assert ann["type"].str.lower().nunique() == ann["type"].nunique(),         "type names collide ignoring case"
    with open(a.index, encoding="utf-8") as fh:
        index = json.load(fh)
    groups = dict(tuple(ann.sort_values("bodyId").groupby("type")))
    os.makedirs(a.out, exist_ok=True)
    wrote = missing = 0
    for row in index["types"]:
        t = row["t"]
        rows = groups.get(t)
        if rows is None or rows.empty:
            missing += 1
            continue
        ids = rows["bodyId"].astype("int64").tolist()
        chosen = TP.pick(ids, MAX_IDS)
        side = rows.set_index("bodyId")["somaSide"].reindex(chosen).astype(object)
        sd = "".join(v if v in ("L", "R") else "o" for v in side.tolist())
        rec = {"t": t, "n": len(ids), "s": TP.subtitle_for(t, rows),
               "ids": [int(i) for i in chosen], "sd": sd}
        with open(os.path.join(a.out, enc(t) + ".json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, separators=(",", ":"))
        wrote += 1
    print("wrote %d preview files to %s (%d index types not in annotations)"
          % (wrote, a.out, missing))


if __name__ == "__main__":
    main()
