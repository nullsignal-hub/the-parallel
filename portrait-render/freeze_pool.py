"""
Freeze the "your neuron" pool into a versioned, immutable file.

Why this exists
---------------
`portrait.py` maps a date to one neuron by hashing the date and indexing into
a pool of eligible neurons. That mapping is only stable as long as *the pool
and its order* are stable. They are not: tightening the eligibility rules
(which we did once already, when unnamed cell types were removed) silently
re-deals every date. A customer who reorders a print would get a different
neuron -- a different artwork under the same description. For a product whose
whole promise is "the same date always returns the same neuron", that is not a
bug in the drawing, it is a broken promise.

So the pool is frozen. A frozen pool file is:

  * an ordered list of bodyIds -- the order *is* the mapping, so it is part of
    the artefact, not an implementation detail;
  * canonically ordered by ascending bodyId, so the file can be rebuilt from
    the same inputs without depending on the row order the annotation file
    happened to ship with;
  * checksummed, so a corrupted or edited pool fails loudly instead of quietly
    shifting everybody's neuron by one;
  * versioned, and the version is printed on the artwork.

Operating rule (also in README.md): pool files are append-only in the same
sense the ledger is. Never edit or regenerate `pool-v1.json`. If the rules
need to change, write `pool-v2.json` and sell it as a new edition; v1 keeps
serving reorders of v1 prints forever.

Usage:
    python freeze_pool.py                 # write pool-v1.json if absent
    python freeze_pool.py --version v2    # cut a new edition
    python freeze_pool.py --verify        # check every pool file on disk
"""
import os
import re
import sys
import json
import hashlib
import argparse
import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANNOT = os.path.join(HERE, "data", "body-annotations.feather")

# --------------------------------------------------------------------------
# The eligibility rules, stated once. Copied verbatim into the frozen file so
# that the file explains itself without this script.
# --------------------------------------------------------------------------
BAD_STATUS = ("orphan", "partially traced", "leaves", "artifact",
              "out of scope", "unimportant", "hard to trace")

UNUSABLE_TYPE = ("", "nan", "none", "unknown", "na")

RULES = [
    "somaLocation is present (the cell body has a known position)",
    "type is present and is not a placeholder string "
    "(%s)" % "/".join(repr(s) for s in UNUSABLE_TYPE if s),
    "statusLabel does not contain any of: %s" % ", ".join(BAD_STATUS),
]

SOURCE = {
    "dataset": "MaleCNS v1.0",
    "file": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "licence": "CC BY 4.0",
    "licence_url": "https://creativecommons.org/licenses/by/4.0/",
}


def build_pool(df):
    """Rows a buyer may legitimately be given. Order is *not* meaningful here;
    `freeze` imposes the canonical order."""
    p = df[df["somaLocation"].notna() & df["type"].notna()].copy()
    t = p["type"].astype(str).str.strip().str.lower()
    p = p[~t.isin(UNUSABLE_TYPE)]
    st = p["statusLabel"].astype(str).str.lower()
    bad = np.zeros(len(p), dtype=bool)
    for k in BAD_STATUS:
        bad |= st.str.contains(k, regex=False).to_numpy()
    return p[~bad].reset_index(drop=True)


def _digest(body_ids):
    """Checksum of the ordered sequence. Order-sensitive on purpose: two pools
    with the same members in a different order are different mappings."""
    h = hashlib.sha256()
    for b in body_ids:
        h.update(b"%d\n" % b)
    return h.hexdigest()


def _file_digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pool_path(version):
    return os.path.join(HERE, "pool-%s.json" % version)


def freeze(version="v1", force=False):
    path = pool_path(version)
    if os.path.exists(path) and not force:
        raise SystemExit(
            "%s already exists.\n"
            "A frozen pool is never regenerated -- prints already sold point\n"
            "into it. Cut a new edition instead:  python freeze_pool.py "
            "--version v2" % os.path.basename(path))

    df = pd.read_feather(ANNOT)
    pool = build_pool(df)
    body_ids = sorted(int(b) for b in pool["bodyId"].to_numpy())
    if len(set(body_ids)) != len(body_ids):
        raise SystemExit("duplicate bodyIds in pool -- refusing to freeze")

    doc = {
        "pool_version": version,
        "frozen_on": datetime.date.today().isoformat(),
        "count": len(body_ids),
        "order": "ascending bodyId",
        "sha256": _digest(body_ids),
        "source": dict(SOURCE, sha256=_file_digest(ANNOT)),
        "rules": RULES,
        "mapping": {
            "description": "date -> neuron. index = "
                           "int(sha256('<date>#<attempt>')[:8], 'big') % count; "
                           "attempt starts at 0 and increments only when a "
                           "candidate fails the printability check.",
            "hash": "sha256",
            "encoding": "utf-8",
        },
        "note": "IMMUTABLE. Do not edit, reorder or regenerate. To change the "
                "eligibility rules, publish a new pool version; this one keeps "
                "serving reorders of artworks already sold under it.",
        "body_ids": body_ids,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return path, doc


def load(version="v1"):
    """Return (ordered bodyId array, metadata). Fails loudly on corruption."""
    path = pool_path(version)
    if not os.path.exists(path):
        raise SystemExit(
            "no frozen pool %s. Run:  python freeze_pool.py --version %s"
            % (os.path.basename(path), version))
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    ids = doc["body_ids"]
    if len(ids) != doc["count"]:
        raise SystemExit("%s: count says %d, list has %d"
                         % (os.path.basename(path), doc["count"], len(ids)))
    if _digest(ids) != doc["sha256"]:
        raise SystemExit(
            "%s: checksum mismatch. The pool has been edited or corrupted; "
            "every date would now resolve to a different neuron. Restore the "
            "original file -- do not regenerate it."
            % os.path.basename(path))
    meta = {k: v for k, v in doc.items() if k != "body_ids"}
    return np.asarray(ids, dtype=np.int64), meta


def verify_all():
    ok = True
    found = sorted(f for f in os.listdir(HERE)
                   if re.fullmatch(r"pool-v\d+\.json", f))
    if not found:
        print("no pool files on disk")
        return False
    for f in found:
        v = f[len("pool-"):-len(".json")]
        try:
            ids, meta = load(v)
        except SystemExit as e:                                # noqa: BLE001
            print("FAIL %s: %s" % (f, e))
            ok = False
            continue
        print("OK   %s  %d neurons  frozen %s  sha256 %s..."
              % (f, len(ids), meta["frozen_on"], meta["sha256"][:16]))
        # does the current rule set still produce this pool?
        try:
            live = sorted(int(b) for b in
                          build_pool(pd.read_feather(ANNOT))["bodyId"].to_numpy())
        except Exception:                                      # noqa: BLE001
            continue
        if live != list(int(i) for i in ids):
            only_live = len(set(live) - set(int(i) for i in ids))
            only_frozen = len(set(int(i) for i in ids) - set(live))
            print("     note: today's rules would give a different pool "
                  "(+%d new, -%d gone). That is expected and harmless -- %s "
                  "is what ships." % (only_live, only_frozen, f))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing pool file (you should not)")
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify_all() else 1)

    path, doc = freeze(a.version, force=a.force)
    print("wrote %s" % os.path.basename(path))
    print("  %d neurons, ordered by %s" % (doc["count"], doc["order"]))
    print("  sha256 %s" % doc["sha256"])
    print("  first/last bodyId: %d / %d"
          % (doc["body_ids"][0], doc["body_ids"][-1]))


if __name__ == "__main__":
    main()
