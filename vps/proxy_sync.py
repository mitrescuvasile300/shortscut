#!/usr/bin/env python3
"""Pull the full Webshare proxy list and verify each proxy against YouTube.

Writes SHORTSCUT_YT_PROXY_FILE (one proxy URL per line, verified-good first)
and refreshes the pipeline's proxy state so the next job starts on a proxy
that is known to pass YouTube's bot check. Run periodically (systemd timer).

Env: WEBSHARE_API_KEY (required), SHORTSCUT_YT_PROXY_FILE, SHORTSCUT_PROXY_STATE.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API = "https://proxy.webshare.io/api/v2/proxy/list/?mode=direct&page=1&page_size=100"
TEST_VIDEO = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo", stable
PROXY_FILE = Path(os.environ.get("SHORTSCUT_YT_PROXY_FILE", "/var/lib/shortscut-processing/proxies.txt"))
STATE_FILE = Path(os.environ.get("SHORTSCUT_PROXY_STATE", "/var/lib/shortscut-processing/.proxy_state.json"))
PARALLEL = 10
TIMEOUT = 60


def fetch_proxies(key: str) -> list[str]:
    req = urllib.request.Request(API, headers={"Authorization": f"Token {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    out = []
    for p in data.get("results", []):
        if not p.get("valid", True):
            continue
        out.append(f"http://{p['username']}:{p['password']}@{p['proxy_address']}:{p['port']}")
    return out


def passes_youtube(proxy: str) -> bool:
    cmd = ["yt-dlp", "--proxy", proxy, "--skip-download", "--print", "id", "--no-warnings", TEST_VIDEO]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        return r.returncode == 0 and "jNQXAC9IVRw" in r.stdout
    except Exception:
        return False


def main() -> int:
    key = os.environ.get("WEBSHARE_API_KEY")
    if not key:
        print("WEBSHARE_API_KEY missing", file=sys.stderr)
        return 2
    proxies = fetch_proxies(key)
    print(f"fetched {len(proxies)} proxies from Webshare")
    t0 = time.time()
    with cf.ThreadPoolExecutor(PARALLEL) as ex:
        results = dict(zip(proxies, ex.map(passes_youtube, proxies)))
    good = [p for p in proxies if results[p]]
    bad = [p for p in proxies if not results[p]]
    print(f"youtube-ok: {len(good)}/{len(proxies)} in {time.time()-t0:.0f}s")
    for p in good:
        print("  OK ", p.split("@")[-1])

    PROXY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROXY_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(good + bad) + "\n", encoding="utf-8")
    tmp.replace(PROXY_FILE)

    # Refresh pipeline state: good -> first verified proxy, bad -> the rest (now).
    now = time.time()
    state = {"good": good[0] if good else None, "bad": {p: now for p in bad}}
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
