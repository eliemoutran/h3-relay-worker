#!/usr/bin/env python3
"""
Client for the serverless H3 Relay character-swap endpoint.

  python swap_sls.py --video clip.mp4 --image person.jpg --describe "a woman with long black hair, lip piercing"
  python swap_sls.py --health
  python swap_sls.py --fetch /runpod-volume/relay/outputs/job_xxxx/swap.mp4

Pay as you go: you are billed only for worker seconds. The script prints the execution time
RunPod reports and the cost at the endpoint's GPU rate, plus the queue/cold-start delay.
API key: RUNPOD_API_KEY or ~/.runpod/api_key. Standard library only (ffmpeg optional).
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ENDPOINT_ID = os.environ.get("H3_ENDPOINT_ID", "ENDPOINT_ID_PLACEHOLDER")
RATE_USD_PER_HOUR = float(os.environ.get("H3_RATE_USD_PER_HOUR", "3.49"))  # BLACKWELL_96 serverless flex rate
RUN_LIMIT = 10 * 1024 * 1024  # /run payload limit
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) swap_sls/1.0"}


def api_key():
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not k:
        p = os.path.join(os.path.expanduser("~"), ".runpod", "api_key")
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                k = f.read().strip()
    if not k:
        sys.exit("No API key: set RUNPOD_API_KEY or ~/.runpod/api_key")
    return k


def call(method, url, key, body=None, timeout=120):
    headers = {"Authorization": f"Bearer {key}"}
    headers.update(UA)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} {url}\n{e.read().decode('utf-8', 'replace')[:2000]}")


def prep_video(src, work):
    """24 fps H.264, shrunk until the base64 payload fits the 10 MB /run limit."""
    if not shutil.which("ffmpeg"):
        return src
    os.makedirs(work, exist_ok=True)
    for crf, scale in ((18, None), (23, None), (26, None), (26, 720), (28, 640)):
        dst = os.path.join(work, f"source_crf{crf}{'_' + str(scale) if scale else ''}.mp4")
        vf = ["-vf", f"scale='if(gt(iw,ih),-2,{scale})':'if(gt(iw,ih),{scale},-2)'"] if scale else []
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-r", "24", *vf, "-c:v", "libx264",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", dst], check=True)
        if os.path.getsize(dst) * 1.37 < RUN_LIMIT - 1024 * 1024:
            return dst
    sys.exit("video too large for the 10 MB request limit even after shrinking; trim it shorter")


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def submit_and_wait(base, key, payload, poll):
    sub = call("POST", f"{base}/run", key, payload)
    jid = sub.get("id")
    if not jid:
        sys.exit(f"unexpected submit response: {sub}")
    print(f"job {jid}")
    t0 = time.time()
    last = None
    while True:
        st = call("GET", f"{base}/status/{jid}", key)
        s = st.get("status")
        if s != last:
            print(f"[{time.time() - t0:6.0f}s] {s}")
            last = s
        if s in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return jid, st
        time.sleep(poll)


def fetch_file(base, key, path, dest):
    offset, size = 0, None
    with open(dest, "wb") as f:
        while True:
            jid, st = submit_and_wait(base, key, {"input": {"action": "fetch", "path": path, "offset": offset,
                                                             "length": 7 * 1024 * 1024}}, 5)
            out = st.get("output") or {}
            if st.get("status") != "COMPLETED" or "data_b64" not in out:
                sys.exit(f"fetch failed: {json.dumps(st)[:1000]}")
            f.write(base64.b64decode(out["data_b64"]))
            offset += out["length"]
            size = out["size_bytes"]
            print(f"  fetched {offset / 1e6:.1f} / {size / 1e6:.1f} MB")
            if out.get("eof"):
                break
    return dest


def main():
    ap = argparse.ArgumentParser(description="Serverless MiniMax H3 character swap")
    ap.add_argument("--video")
    ap.add_argument("--image")
    ap.add_argument("--subject", default="woman")
    ap.add_argument("--describe", default="")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--seed", type=int, default=904239)
    ap.add_argument("--window-frames", type=int, default=124)
    ap.add_argument("--history-frames", type=int, default=18)
    ap.add_argument("--resolution", default="768", choices=["768", "640", "512"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--endpoint", default=ENDPOINT_ID)
    ap.add_argument("--health", action="store_true", help="run the worker health action and exit")
    ap.add_argument("--fetch", default=None, metavar="VOLUME_PATH", help="download a large output by its volume path")
    a = ap.parse_args()

    key = api_key()
    base = f"https://api.runpod.ai/v2/{a.endpoint}"
    os.makedirs(a.out, exist_ok=True)

    if a.health:
        jid, st = submit_and_wait(base, key, {"input": {"action": "health"}}, 5)
        print(json.dumps(st.get("output"), indent=1))
        print(f"delay {st.get('delayTime', 0) / 1000:.0f}s  execution {st.get('executionTime', 0) / 1000:.0f}s")
        return
    if a.fetch:
        dest = os.path.join(a.out, (a.name or os.path.basename(os.path.dirname(a.fetch)) or "fetched") + ".mp4")
        fetch_file(base, key, a.fetch, dest)
        print(f"saved {dest}")
        return
    if not a.video or not a.image:
        sys.exit("--video and --image are required")
    for p in (a.video, a.image):
        if not os.path.isfile(p):
            sys.exit(f"file not found: {p}")

    name = a.name or time.strftime("swap_%Y%m%d-%H%M%S")
    video = prep_video(a.video, os.path.join(a.out, name + "_inputs"))
    job_input = {
        "video": b64(video), "image": b64(a.image), "subject": a.subject, "seed": a.seed,
        "build_sheet": not a.no_sheet, "window_frames": a.window_frames, "history_frames": a.history_frames,
        "resolution": a.resolution,
    }
    if a.describe:
        job_input["describe"] = a.describe
    if a.prompt:
        job_input["prompt"] = a.prompt
    payload = {"input": job_input}
    size = len(json.dumps(payload).encode())
    print(f"payload {size / 1e6:.2f} MB  (video {os.path.getsize(video) / 1e6:.2f} MB)")

    jid, st = submit_and_wait(base, key, payload, a.poll)
    out = st.get("output") or {}
    exec_s = (st.get("executionTime") or 0) / 1000
    delay_s = (st.get("delayTime") or 0) / 1000
    cost = exec_s / 3600 * RATE_USD_PER_HOUR
    if st.get("status") != "COMPLETED" or "error" in out:
        print(json.dumps({k: v for k, v in out.items() if k != "video_b64"}, indent=1)[:4000])
        print(f"execution {exec_s:.0f}s billed anyway (${cost:.3f})")
        sys.exit("swap failed")

    dest = os.path.join(a.out, name + ".mp4")
    if out.get("video_b64"):
        with open(dest, "wb") as f:
            f.write(base64.b64decode(out["video_b64"]))
    else:
        print(out.get("note", "large output, fetching in chunks"))
        fetch_file(base, key, out["path"], dest)
    out_s = out.get("seconds")
    report = {k: v for k, v in out.items() if k != "video_b64"}
    report.update({"local_output": dest, "job_id": jid, "execution_seconds": round(exec_s, 1),
                   "queue_or_coldstart_seconds": round(delay_s, 1), "rate_usd_per_hour": RATE_USD_PER_HOUR,
                   "cost_usd": round(cost, 3),
                   "cost_per_output_second_usd": round(cost / out_s, 4) if out_s else None})
    with open(os.path.join(a.out, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"saved  {dest}")
    line = f"billed {exec_s / 60:.1f} min  ->  ${cost:.3f} at ${RATE_USD_PER_HOUR}/h"
    if out_s:
        line += f"  (${cost / out_s:.3f} per output second, {out_s:.1f}s clip)"
    print(line)
    if delay_s > 60:
        print(f"note   {delay_s / 60:.1f} min of queue/cold start before execution (a cold worker loads models first)")


if __name__ == "__main__":
    main()
