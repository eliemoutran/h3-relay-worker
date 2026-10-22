"""
RunPod Serverless handler: MiniMax H3 character swap via akatz H3 Relay 2.0.

Job input
  video        base64 or http(s) URL of the source clip (one continuous shot works best)
  image        base64 or http(s) URL of the reference photo (or a finished character sheet)
  subject      who to replace, for SAM 3.1 ("woman", "man in the red shirt"); default "woman"
  prompt       full prompt (optional); default is the author's prompt, or built from `describe`
  describe     short description of the reference character (optional, recommended)
  build_sheet  true (default) builds a two-view sheet (full + head crop) from `image`
  head_box     [x, y, w, h] head crop in the reference, optional
  seed         int, default 904239
  window_frames 124, history_frames 18, resolution "768"|"640"|"512", activation_gib 17.0
  action       "health" -> status only; "fetch" with path/offset/length -> base64 chunk of a volume file

Output
  video_b64 (when <= INLINE_MAX_MB), path (always, on the network volume), size_bytes,
  seconds (output), job_seconds, prepared (ComfyUI's prepared size/frames), prompt_id
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid

import runpod

V = os.environ.get("RELAY_ROOT", "/runpod-volume/relay")
COMFY_DIR = os.path.join(V, "ComfyUI")
COMFY = "http://127.0.0.1:8188"
OUT_ROOT = os.path.join(V, "outputs")
GRAPH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swap_prompt.json")
INLINE_MAX = int(os.environ.get("INLINE_MAX_MB", "10")) * 1024 * 1024
DEFAULT_PROMPT = ("Replace the selected person with the reference character. "
                  "Preserve source performance, scene and audio timing.")
os.environ["PATH"] = os.path.join(V, "bin") + os.pathsep + os.environ.get("PATH", "")

_comfy_proc = None


def log(msg):
    print(f"[handler] {msg}", flush=True)


def http(path, data=None, timeout=60):
    req = urllib.request.Request(COMFY + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if resp.headers.get("Content-Type", "").startswith("application/json") else raw


def comfy_alive():
    try:
        http("/system_stats", timeout=5)
        return True
    except Exception:
        return False


def ensure_comfy(timeout_s=900):
    global _comfy_proc
    if comfy_alive():
        return
    if _comfy_proc is None or _comfy_proc.poll() is not None:
        log("starting ComfyUI")
        _comfy_proc = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"],
            cwd=COMFY_DIR, stdout=sys.stdout, stderr=sys.stderr)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if comfy_alive():
            log(f"ComfyUI ready in {time.time() - t0:.0f}s")
            return
        if _comfy_proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with code {_comfy_proc.returncode}")
        time.sleep(2)
    raise RuntimeError("ComfyUI did not come up")


def media_bytes(value):
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=300) as r:
            return r.read()
    if isinstance(value, str):
        if "," in value and value.strip().startswith("data:"):
            value = value.split(",", 1)[1]
        return base64.b64decode(value)
    raise ValueError("media must be a base64 string or an http(s) URL")


def to_24fps(src, dst):
    """Re-encode to 24 fps H.264 when ffmpeg is available; otherwise copy as-is (the Relay
    prepare node normalizes the frame rate itself, this just makes the input predictable)."""
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-r", "24", "-c:v", "libx264", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", dst], check=True)
            return dst
        except subprocess.CalledProcessError as e:
            log(f"ffmpeg re-encode failed ({e}); using the source as uploaded")
    shutil.copy(src, dst)
    return dst


def probe_duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=noprint_wrappers=1:nokey=1", path], capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        return None


def build_sheet(src, dst, head_box=None):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    width, height = im.size
    if head_box:
        x, y, w, h = head_box
    else:
        w = h = int(min(width, height) * 0.7)
        x = (width - w) // 2
        y = int(height * 0.05)
    crop = im.crop((x, y, x + w, y + h)).resize((1000, 1000), Image.LANCZOS)
    full = im.resize((max(32, int(width * 1000 / height)), 1000), Image.LANCZOS)
    pad = 60
    sheet = Image.new("RGB", (full.width + crop.width + pad * 3, 1000 + pad * 2), "white")
    sheet.paste(full, (pad, pad))
    sheet.paste(crop, (full.width + pad * 2, pad))
    sheet.save(dst)
    return dst


def prune_outputs(max_age_s=7 * 86400):
    try:
        cutoff = time.time() - max_age_s
        for d in os.listdir(OUT_ROOT):
            p = os.path.join(OUT_ROOT, d)
            if os.path.isdir(p) and os.stat(p).st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def run_swap(job):
    inp = job["input"]
    job_id = job.get("id") or uuid.uuid4().hex[:8]
    tag = f"job_{job_id[:8]}"
    in_dir = os.path.join(COMFY_DIR, "input")
    os.makedirs(in_dir, exist_ok=True)
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    # inputs
    raw_video = os.path.join(out_dir, "source_raw.mp4")
    with open(raw_video, "wb") as f:
        f.write(media_bytes(inp["video"]))
    video = to_24fps(raw_video, os.path.join(in_dir, f"{tag}_source.mp4"))
    src_dur = probe_duration(video)
    ref_raw = os.path.join(out_dir, "reference_raw.png")
    with open(ref_raw, "wb") as f:
        f.write(media_bytes(inp["image"]))
    if inp.get("build_sheet", True):
        ref = build_sheet(ref_raw, os.path.join(in_dir, f"{tag}_reference.png"), inp.get("head_box"))
    else:
        from PIL import Image
        ref = os.path.join(in_dir, f"{tag}_reference.png")
        Image.open(ref_raw).convert("RGB").save(ref)
    shutil.copy(ref, os.path.join(out_dir, "reference_used.png"))

    describe = (inp.get("describe") or "").strip()
    if inp.get("prompt"):
        prompt = inp["prompt"]
    elif describe:
        prompt = (f"Replace the selected person with the reference character shown in the character sheet: "
                  f"{describe}. Keep the face, hair and skin exactly as in the reference. "
                  f"Preserve the source performance, outfit, scene and audio timing.")
    else:
        prompt = DEFAULT_PROMPT

    with open(GRAPH, "r", encoding="utf-8") as f:
        g = json.load(f)
    g["source"]["inputs"]["file"] = os.path.basename(video)
    g["reference"]["inputs"]["image"] = os.path.basename(ref)
    g["subject"]["inputs"]["text"] = inp.get("subject") or "woman"
    g["reference_conditioning"]["inputs"]["prompt"] = prompt
    g["plan"]["inputs"]["seed"] = int(inp.get("seed", 904239))
    g["plan"]["inputs"]["window_frames"] = str(int(inp.get("window_frames", 124)))
    g["plan"]["inputs"]["history_frames"] = str(int(inp.get("history_frames", 18)))
    g["prepare"]["inputs"]["resolution"] = str(inp.get("resolution", "768"))
    g["sampling_memory"]["inputs"]["activation_memory_gib"] = float(inp.get("activation_gib", 17.0))
    g["save"]["inputs"]["filename_prefix"] = f"h3-relay/{tag}"

    ensure_comfy()
    res = http("/prompt", json.dumps({"prompt": g, "client_id": tag}).encode())
    if "error" in res or res.get("node_errors"):
        return {"error": "prompt rejected", "detail": res}
    pid = res["prompt_id"]
    log(f"queued {pid} source={src_dur}s")
    t0 = time.time()
    while True:
        h = http(f"/history/{pid}")
        if pid in h:
            break
        time.sleep(5)
    entry = h[pid]
    st = entry.get("status", {})
    if st.get("status_str") != "success":
        err = [m[1] for m in st.get("messages", []) if m[0] == "execution_error"]
        msg = err[0].get("exception_message") if err else "unknown"
        return {"error": "execution failed", "exception": msg, "prompt_id": pid,
                "node": err[0].get("node_type") if err else None}
    msgs = {m[0]: m[1] for m in st.get("messages", [])}
    job_s = time.time() - t0
    if "execution_start" in msgs and "execution_success" in msgs:
        job_s = (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000

    # locate the saved video
    saved = None
    for nid, out in entry.get("outputs", {}).items():
        for key, items in out.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("type") == "output" and str(item.get("filename", "")).lower().endswith((".mp4", ".webm", ".mov")):
                    saved = os.path.join(COMFY_DIR, "output", item.get("subfolder", ""), item["filename"])
    if not saved or not os.path.isfile(saved):
        return {"error": "no video output", "outputs": entry.get("outputs"), "prompt_id": pid}
    final = os.path.join(out_dir, "swap.mp4")
    shutil.move(saved, final)
    size = os.path.getsize(final)
    result = {
        "path": final, "size_bytes": size, "seconds": probe_duration(final), "source_seconds": src_dur,
        "job_seconds": round(job_s, 1), "prompt": prompt, "subject": g["subject"]["inputs"]["text"],
        "seed": g["plan"]["inputs"]["seed"], "prompt_id": pid,
    }
    if size <= INLINE_MAX:
        with open(final, "rb") as f:
            result["video_b64"] = base64.b64encode(f.read()).decode()
    else:
        result["note"] = f"{size / 1e6:.1f} MB exceeds the inline limit; download with action=fetch in chunks"
    # keep ComfyUI's input dir clean
    for p in (video, ref):
        try:
            os.remove(p)
        except OSError:
            pass
    prune_outputs()
    return result


def fetch_chunk(inp):
    path = inp.get("path", "")
    if not os.path.abspath(path).startswith(os.path.abspath(OUT_ROOT)):
        return {"error": "path must be under the outputs folder"}
    if not os.path.isfile(path):
        return {"error": "file not found", "path": path}
    size = os.path.getsize(path)
    offset = int(inp.get("offset", 0))
    length = int(inp.get("length", 7 * 1024 * 1024))
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(length)
    return {"path": path, "size_bytes": size, "offset": offset, "length": len(data),
            "eof": offset + len(data) >= size, "data_b64": base64.b64encode(data).decode()}


def handler(job):
    inp = job.get("input") or {}
    action = inp.get("action")
    if action == "health":
        try:
            ensure_comfy()
            stats = http("/system_stats")
            return {"status": "ok", "comfyui": stats["system"].get("comfyui_version"),
                    "gpu": stats["devices"][0]["name"], "vram_gib": round(stats["devices"][0]["vram_total"] / 2**30, 1),
                    "models_gib": round(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(COMFY_DIR, "models")) for f in fs) / 2**30, 1)}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    if action == "fetch":
        return fetch_chunk(inp)
    if "video" not in inp or "image" not in inp:
        return {"error": "input needs 'video' and 'image' (base64 or URL)"}
    try:
        return run_swap(job)
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()[-3000:]}


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    try:
        ensure_comfy()
    except Exception as e:
        log(f"ComfyUI prestart failed: {e}")
    runpod.serverless.start({"handler": handler})
