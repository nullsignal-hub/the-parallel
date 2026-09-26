"""
LAB sample sheets: the 20 SAMPLE_TYPES (build_types_index.py) at low
resolution, one style, plus a contact sheet _contact.png.

    python lab_samples.py --out-dir <dir> [--style ink] [--size 18x24] [--dpi 60]
    python lab_samples.py --out-dir <dir> --styles ink,spectral,blueprint --no-contact
        (sweep: every sample type x every listed style; errors are counted)

No ranking or aesthetic judgement is made here -- it only produces the images.
"""
import os
import sys
import json
import time
import argparse
import traceback

from PIL import Image, ImageDraw, ImageFont

import lab_inputs as L
from build_types_index import SAMPLE_TYPES
from type_poster import render_type


def safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def contact_sheet(paths, labels, out, cols=5, cell=360):
    thumbs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell, cell))
        thumbs.append(im)
    rows = (len(thumbs) + cols - 1) // cols
    pad, lab = 12, 22
    W = cols * (cell + pad) + pad
    H = rows * (cell + lab + pad) + pad
    sheet = Image.new("RGB", (W, H), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for i, (im, text) in enumerate(zip(thumbs, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + lab + pad)
        sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        d.text((x, y + cell + 3), text, fill=(230, 230, 230), font=font)
    sheet.save(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--styles", default="ink")
    ap.add_argument("--size", default="18x24", choices=list(L.LAB_SIZES))
    ap.add_argument("--dpi", type=int, default=60)
    ap.add_argument("--caption", default="Your lab name  ·  SfN 2026")
    ap.add_argument("--no-contact", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    styles = a.styles.split(",")
    names = L.load_type_names()
    results, errors = [], []
    for style in styles:
        for t, why in SAMPLE_TYPES:
            out = os.path.join(a.out_dir, "%02d_%s_%s.png" % (
                [x for x, _ in SAMPLE_TYPES].index(t) + 1, safe(t), style))
            try:
                info = render_type(t, style, a.size, a.caption, dpi=a.dpi,
                                   outfile=out, type_names=names)
                info["why"] = why
                results.append(info)
                print(json.dumps({k: info[k] for k in ("type", "style", "paper_in", "n_type",
                                                       "n_drawn", "n_outliers", "bytes",
                                                       "seconds_total")}), flush=True)
            except BaseException as e:                      # noqa: BLE001
                if isinstance(e, KeyboardInterrupt):
                    raise
                errors.append({"type": t, "style": style, "error": repr(e)})
                traceback.print_exc()
    if not a.no_contact and results:
        first = [r for r in results if r["style"] == styles[0]]
        contact_sheet([r["out"] for r in first],
                      ["%s (%d)" % (r["type"], r["n_type"]) for r in first],
                      os.path.join(a.out_dir, "_contact.png"))
    with open(os.path.join(a.out_dir, "_samples.json"), "w", encoding="utf-8") as fh:
        json.dump({"results": results, "errors": errors}, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"rendered": len(results), "errors": len(errors)}))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
