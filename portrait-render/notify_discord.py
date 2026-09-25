"""
Seller alert for render-neuron.yml, same embed shape as paypal-worker's
notifyDiscord(). Silently skips when DISCORD_WEBHOOK_URL is unset, and never
fails the run (an alert failing must not change the outcome).

Environment: DISCORD_WEBHOOK_URL, URGENT ("1" pings @everyone), TITLE,
ORDER_ID, STYLE, BODY_ID, CELL_TYPE, DRAFT_ID, FILE_URL, RUN_URL, NEXT.
"""
import datetime
import json
import os
import urllib.request


def build_payload(env):
    urgent = env.get("URGENT") == "1"
    title = env.get("TITLE") or "render-neuron"
    g = lambda k: env.get(k) or ""                                  # noqa: E731
    fields = [
        ("PayPal order", g("ORDER_ID"), True),
        ("Style", g("STYLE"), True),
        ("bodyId / type", "%s / %s" % (g("BODY_ID"), g("CELL_TYPE") or "-"), True),
        ("Printful draft", ("#" + g("DRAFT_ID")) if g("DRAFT_ID") else "-", True),
        ("Reference", "neuron-%s-%s-%s" % (g("BODY_ID"), g("STYLE"), g("ORDER_ID")), False),
        ("Print file", g("FILE_URL"), False),
        ("Run", g("RUN_URL"), False),
        ("Next step", g("NEXT"), False),
    ]
    return {
        "content": ("@everyone " + title) if urgent else title,
        "allowed_mentions": {"parse": ["everyone"] if urgent else []},
        "embeds": [{
            "title": title[:250],
            "color": 0xd9603a if urgent else 0xc9a227,
            "fields": [{"name": n[:250], "value": (v or "-")[:1000], "inline": i}
                       for n, v, i in fields],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }],
    }


def main():
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        print("DISCORD_WEBHOOK_URL not set; skipping")
        return
    req = urllib.request.Request(
        url, data=json.dumps(build_payload(os.environ)).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "the-parallel-render-neuron"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print("discord", r.status)
    except Exception as ex:                                        # noqa: BLE001
        print("discord notify failed:", ex)


if __name__ == "__main__":
    main()
