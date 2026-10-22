#!/bin/bash
# Cold-start bootstrap for the H3 Relay character-swap worker.
#
# Runs as the container command on RunPod Serverless using the stock
# runpod/pytorch image. Layout:
#   network volume (/runpod-volume/relay): ComfyUI checkout, h3-relay nodes, models,
#                                          static ffmpeg, a pip wheel cache, outputs
#   container disk (/opt/venv):            the Python env, rebuilt from the wheel cache
#                                          on every cold start (~1-2 min, local NVMe)
# Python packages are deliberately NOT installed onto the network volume: pip writing
# thousands of small files over network storage stalls for tens of minutes.
set -uo pipefail
V=${RELAY_ROOT:-/runpod-volume/relay}
REPO=${WORKER_REPO:-https://github.com/eliemoutran/h3-relay-worker.git}
COMFY_COMMIT=${COMFY_COMMIT:-79be670e2d9be63e238785af307369d2b9039ed1}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
VENV=/opt/venv
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

mkdir -p "$V" && cd "$V"
log "start (volume root $V)"
# one installer per volume: a second worker booting on the same volume waits here
exec 9>"$V/.bootstrap.lock"
if ! flock -n 9; then log "another worker is installing on this volume; waiting"; flock 9; log "lock acquired"; fi
rm -rf "$V/venv" 2>/dev/null   # leftover from the earlier volume-hosted env

# 1. worker code, always refreshed so handler fixes land without a rebuild
if [ -d worker/.git ]; then
  git -C worker pull -q --ff-only || log "worker pull failed, using existing checkout"
else
  git clone -q "$REPO" worker || { log "FATAL: cannot clone $REPO"; exit 1; }
fi

# 2. static ffmpeg/ffprobe on the volume (the image has none)
if [ ! -x bin/ffmpeg ] || [ ! -x bin/ffprobe ]; then
  log "fetching static ffmpeg"
  mkdir -p bin && curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    | tar -xJ --no-same-owner --no-same-permissions --strip-components=1 -C bin --wildcards '*/ffmpeg' '*/ffprobe'
  chmod +x bin/ffmpeg bin/ffprobe 2>/dev/null
  [ -x bin/ffmpeg ] && log "ffmpeg ok: $(bin/ffmpeg -version | head -1)" || log "WARN: ffmpeg unavailable; handler will skip local re-encode"
fi
export PATH="$V/bin:$PATH"

# 3. ComfyUI at the qualified commit + h3-relay (git checkouts live fine on the volume)
if [ ! -d ComfyUI/.git ]; then log "cloning ComfyUI"; git clone -q https://github.com/comfyanonymous/ComfyUI.git; fi
if [ "$(git -C ComfyUI rev-parse HEAD)" != "$COMFY_COMMIT" ]; then git -C ComfyUI fetch -q origin && git -C ComfyUI checkout -q "$COMFY_COMMIT"; fi
[ -d ComfyUI/custom_nodes/h3-relay ] || { log "cloning h3-relay"; git clone -q https://github.com/akatz-ai/h3-relay.git ComfyUI/custom_nodes/h3-relay; }

# 4. wheel cache on the volume (large sequential files: fast), built once
WHEELS="$V/wheels"
if [ ! -f "$WHEELS/.ok" ]; then
  log "downloading wheels to the volume cache (one-time)"
  mkdir -p "$WHEELS"
  pip download -q -d "$WHEELS" torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      -r ComfyUI/requirements.txt runpod pillow requests --extra-index-url "$TORCH_INDEX" \
    && touch "$WHEELS/.ok" || { log "FATAL: wheel download failed"; exit 1; }
  log "wheel cache: $(du -sh "$WHEELS" | cut -f1)"
fi

# 5. local venv from the cache (every cold start; local NVMe)
log "building local venv"
python -m venv "$VENV" || { pip install -q virtualenv && python -m virtualenv "$VENV"; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --no-index --find-links "$WHEELS" torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    -r ComfyUI/requirements.txt runpod pillow requests || { log "FATAL: venv install from cache failed"; exit 1; }
python -c "import torch, comfy_kitchen; print('[bootstrap] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'kitchen', comfy_kitchen.__version__)" 2>/dev/null \
  || python -c "import torch; print('[bootstrap] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 6. models (parallel, resumable, skipped once present)
cd ComfyUI
dl() { d="$1"; f="$2"; u="$3"; mkdir -p "models/$d"
  if [ -s "models/$d/$f" ]; then return 0; fi
  log "downloading $f"; curl -sL -C - --retry 3 -o "models/$d/$f.part" "$u" && mv "models/$d/$f.part" "models/$d/$f" || { log "DOWNLOAD FAILED $f"; return 1; }; }
H=https://huggingface.co
dl diffusion_models minimax_h3_ref2va_pruned_int8_convrot.safetensors $H/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors &
dl text_encoders qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors $H/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors &
dl vae minimax_h3_video_vae_fp16.safetensors $H/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors &
dl vae minimax_h3_audio_vae_fp32.safetensors $H/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors &
dl checkpoints sam3.1_multiplex_fp16.safetensors $H/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors &
dl loras minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors $H/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors &
dl loras/h3-v2-challenge-eval-20261013 v2-1000.safetensors $H/akatz-ai/MiniMax-H3-Character-Swap-LoRA/resolve/main/h3_character_swap_v1.5.safetensors &
wait
for f in models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
         models/vae/minimax_h3_video_vae_fp16.safetensors models/vae/minimax_h3_audio_vae_fp32.safetensors models/checkpoints/sam3.1_multiplex_fp16.safetensors \
         models/loras/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors models/loras/h3-v2-challenge-eval-20261013/v2-1000.safetensors; do
  [ -s "$f" ] || { log "FATAL: missing $f"; exit 1; }
done
log "models ready: $(du -sh models | cut -f1)"

# 7. hand over to the serverless handler (it starts ComfyUI itself)
cd "$V/worker"
log "starting handler"
exec python handler.py
