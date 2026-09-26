"""
LAB (cell-type poster for labs) -- input rules shared by type_poster.py and
.github/workflows/render-type.yml.

Standard library only, so the workflow can validate its inputs before
setup-python / pip run (and a failure still reaches the @everyone alert).

The same rules are implemented in the checkout Worker (paypal-worker/worker.js,
labCaptionError / LAB_SIZES / LAB_STYLES). Change both together.

    python lab_inputs.py --type EPG --style ink --size 18x24 --caption "..."
        -> exit 0 and prints the normalised caption as JSON, or exit 1.
"""
import os
import re
import sys
import json
import argparse
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES_INDEX = os.path.join(HERE, "types-index.json")

LAB_STYLES = ("ink", "spectral", "blueprint", "duotone", "constellation")
# paper sizes on sale -> (short side, long side) in inches; the sheet is
# turned to match the subject by render._orient.
LAB_SIZES = {"18x24": (18.0, 24.0), "24x36": (24.0, 36.0)}
CAPTION_MAX = 60

# A type name as it appears in the MaleCNS annotations. Membership in
# types-index.json is the real check; this only keeps shell/file-name-hostile
# characters out before the index is even opened.
TYPE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-.()+/ ,'&]{1,64}$")

# Caption: what the poster font (DejaVu Sans) actually has glyphs for, so a
# caption can never print as empty boxes. Latin (ASCII + Latin-1 + Latin
# Extended-A, for lab / PI names with accents), Greek (MBON-γ1pedc, α/β
# lobes), and a little typographic punctuation. No CJK: the print font has
# no Japanese glyphs (see IMPLEMENTATION-LOG-lab-backend.md).
_CAPTION_OK = re.compile(
    "^["
    " -~"          # printable ASCII
    "¡-¬®-ſ"   # Latin-1 letters/signs (minus soft hyphen) + Latin Ext-A
    "Ͱ-ͳͶ-ͷͻ-ͽ΄-ΊΌΎ-ΡΣ-Ͽ"  # Greek
    "–—‘’“”…′″"  # – — ‘ ’ “ ” … ′ ″
    "]*$")


def normalise_caption(raw):
    """(caption, error). caption is NFC, trimmed, inner whitespace runs of
    plain spaces kept as-is. Empty / None -> ("", None)."""
    if raw is None:
        return "", None
    if not isinstance(raw, str):
        return None, "caption must be a string"
    s = unicodedata.normalize("NFC", raw).strip(" ")
    if not s:
        return "", None
    if len(s) > CAPTION_MAX:
        return None, "caption longer than %d characters" % CAPTION_MAX
    if not _CAPTION_OK.match(s):
        return None, ("caption may only use Latin letters, digits, Greek and "
                      "basic punctuation (no control characters, no CJK)")
    return s, None


def load_type_names(path=TYPES_INDEX):
    with open(path, encoding="utf-8") as fh:
        idx = json.load(fh)
    return {t["t"] for t in idx["types"]}


def validate(cell_type, style, size, caption, type_names=None):
    """Returns (normalised caption, [errors])."""
    errs = []
    if not isinstance(cell_type, str) or not TYPE_NAME_RE.match(cell_type):
        errs.append("bad type")
    else:
        names = type_names if type_names is not None else load_type_names()
        if cell_type not in names:
            errs.append("type not in types-index.json")
    if style not in LAB_STYLES:
        errs.append("bad style")
    if size not in LAB_SIZES:
        errs.append("bad size")
    cap, cerr = normalise_caption(caption)
    if cerr:
        errs.append(cerr)
    return cap, errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--caption", default="")
    ap.add_argument("--index", default=TYPES_INDEX)
    a = ap.parse_args(argv)
    cap, errs = validate(a.type, a.style, a.size, a.caption,
                         load_type_names(a.index))
    if errs:
        print("; ".join(errs), file=sys.stderr)
        return 1
    print(json.dumps({"type": a.type, "style": a.style, "size": a.size,
                      "caption": cap}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
