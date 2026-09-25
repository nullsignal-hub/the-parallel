"""
Place a DRAFT Printful order for one rendered neuron sheet.
Used by .github/workflows/render-neuron.yml (step "Create draft Printful order").

Environment:
  PRINTFUL_TOKEN   store-scoped private token ("PARALLEL SELVES -- Direct Checkout")
  BODY_ID, STYLE, ORDER_ID, CELL_TYPE
  RECIPIENT_B64    base64 JSON {name,address1,address2,city,state_code,country_code,zip,email}
  FILE_URL         public URL of the 3600x3600 PNG
  PRINTFUL_API     optional, default https://api.printful.com (tests point this at a mock)
  GITHUB_OUTPUT    optional; draft_id= / external_id= are appended when set

external_id is "neuron-<ORDER_ID>". Printful caps external_id at 32
characters (digits, letters, dashes, underscores), which the longer
"neuron-<bodyId>-<style>-<orderID>" reference always exceeds with a 17-char
PayPal id; that reference goes into the item name instead. The checkout
Worker keys its KV order record by the same external_id.

No "confirm" is sent: the order stays a draft until the seller confirms it.
Retries 429 / 5xx / network errors (3 attempts). "already exists" resolves
to the existing order (an earlier attempt or an earlier run created it).
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

VARIANT_ID = 4464            # Enhanced Matte Paper Poster (in), 12"x12"
MAX_EXTERNAL_ID = 32


def external_id_for(order_id):
    return "neuron-" + order_id


def build_order(env):
    external_id = external_id_for(env["ORDER_ID"])
    if len(external_id) > MAX_EXTERNAL_ID:
        raise SystemExit("external_id longer than Printful's %d-character limit: %s"
                         % (MAX_EXTERNAL_ID, external_id))
    recipient = json.loads(base64.b64decode(env["RECIPIENT_B64"]).decode("utf-8"))
    for k in ("name", "address1", "country_code", "zip"):
        if not recipient.get(k):
            raise SystemExit("recipient is missing " + k)
    name = "PARALLEL SELVES - %s - bodyId %s - %s" % (
        env.get("CELL_TYPE") or "neuron", env["BODY_ID"], env["STYLE"])
    return {
        "external_id": external_id,
        "recipient": recipient,
        "items": [{
            "variant_id": VARIANT_ID,
            "quantity": 1,
            "name": name[:120],
            "files": [{"type": "default", "url": env["FILE_URL"]}],
        }],
    }


def _call(api, token, method, path, payload=None):
    req = urllib.request.Request(
        api + path, method=method,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + token,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except ValueError:
            return e.code, {}


def place(env, sleep=time.sleep):
    api = env.get("PRINTFUL_API") or "https://api.printful.com"
    token = env["PRINTFUL_TOKEN"]
    order = build_order(env)
    ext = order["external_id"]
    last = None
    for attempt in range(3):
        if attempt:
            sleep(3 * attempt)
        try:
            status, data = _call(api, token, "POST", "/orders", order)
        except (urllib.error.URLError, OSError) as e:
            last = "network: %s" % e
            continue
        if 200 <= status < 300:
            return (data.get("result") or {}).get("id"), ext
        text = json.dumps(data)
        last = "HTTP %s %s" % (status, text[:500])
        if "already exists" in text.lower():
            try:
                s2, ex = _call(api, token, "GET", "/orders/@" + ext)
            except (urllib.error.URLError, OSError) as e:
                s2, ex = 0, {}
                last += " / lookup network: %s" % e
            oid = (ex.get("result") or {}).get("id") if 200 <= s2 < 300 else None
            if oid:
                return oid, ext
        if status != 429 and status < 500:
            break
    raise SystemExit("Printful draft order failed: %s" % last)


def main():
    draft_id, ext = place(os.environ)
    print("Printful draft order", draft_id, "external_id", ext)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write("draft_id=%s\nexternal_id=%s\n" % (draft_id, ext))


if __name__ == "__main__":
    main()
