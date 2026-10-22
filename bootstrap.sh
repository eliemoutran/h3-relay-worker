#!/bin/bash
# Cold-start bootstrap for the H3 Relay character-swap worker.
#
# Runs as the container command on RunPod Serverless using the stock
# runpod/pytorch image. Everything heavy lives on the network volume
# (/runpod-volume/relay): a Python venv, ComfyUI pinned to the H3 Relay
# qualified commit, the h3-relay custom nodes, a static ffmpeg and the models.
# The first boot on a fresh volume installs all of it (~10 minutes); later cold
# starts only activate the venv, refresh this repo and start ComfyUI.
set -uo pipefail
V=${RELAY_ROOT:-/runpod-volume/relay}
REPO=${WORKER_REPO:-https://github.com/eliemoutran/h3-relay-worker.git}
COMFY_COMMIT=${COMFY_COMMIT:-79be670e2d9be63e238785af307369d2b9039ed1}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

mkdir -p "$V" && cd "$V"
log "start (volume root $V)"

# 1. worker code, always refreshed so handler fixes land without a rebuild
if [ -d worker/.git ]; then
  git -C worker pull -q --ff-only || log "worker pull failed, using existing checkout"
else
  git clone -q "$REPO" worker || { log "FATAL: cannot clone $REPO"; exit 1; }
fi

# 2. static ffmpeg/ffprobe on the volume (the image has none)
if [ ! -x bin/ffmpeg ]; then
  log "fetching static ffmpeg"
  mkdir -p bin && curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    | tar -xJ --strip-components=1 -C bin --wildcards '*/ffmpeg' '*/ffprobe' || log "WARN: ffmpeg download failed"
fi
export PATH="$V/bin:$PATH"

# 3. python venv on the volume
if [ ! -x venv/bin/python ]; then
  log "creating venv"
  python -m venv venv || { pip install -q virtualenv && python -m virtualenv venv; }
fi
# shellcheck disable=SC1091
source venv/bin/activate
if [ ! -f .torch_ok ]; then
  log "installing torch 2.11 ($TORCH_INDEX)"
  pip install -q torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url "$TORCH_INDEX" && touch .torch_ok
fi

# 4. ComfyUI at the qualified commit + h3-relay
if [ ! -d ComfyUI/.git ]; then log "cloning ComfyUI"; git clone -q https://github.com/comfyanonymous/ComfyUI.git; fi
cd ComfyUI
if [ "$(git rev-parse HEAD)" != "$COMFY_COMMIT" ]; then git fetch -q origin && git checkout -q "$COMFY_COMMIT"; fi
if [ ! -f "$V/.req_ok" ]; then
  log "installing ComfyUI requirements"
  pip install -q -r requirements.txt && pip install -q runpod pillow requests && touch "$V/.req_ok"
fi
[ -d custom_nodes/h3-relay ] || { log "cloning h3-relay"; git clone -q https://github.com/akatz-ai/h3-relay.git custom_nodes/h3-relay; }

# 5. models (resumable, skipped once present)
dl() { d="$1"; f="$2"; u="$3"; mkdir -p "models/$d"
  if [ -s "models/$d/$f" ]; then return 0; fi
  log "downloading $f"; curl -sL -C - --retry 3 -o "models/$d/$f" "$u" || { log "DOWNLOAD FAILED $f"; rm -f "models/$d/$f"; exit 1; }; }
H=https://huggingface.co
dl diffusion_models minimax_h3_ref2va_pruned_int8_convrot.safetensors $H/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
dl text_encoders qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors $H/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
dl vae minimax_h3_video_vae_fp16.safetensors $H/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors
dl vae minimax_h3_audio_vae_fp32.safetensors $H/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors
dl checkpoints sam3.1_multiplex_fp16.safetensors $H/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors
dl loras minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors $H/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors
dl loras/h3-v2-challenge-eval-20261013 v2-1000.safetensors $H/akatz-ai/MiniMax-H3-Character-Swap-LoRA/resolve/main/h3_character_swap_v1.5.safetensors
log "models ready: $(du -sh models | cut -f1)"

python -c "import torch, comfy_kitchen; print('[bootstrap] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'kitchen', comfy_kitchen.__version__)" 2>/dev/null \
  || python -c "import torch; print('[bootstrap] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 6. hand over to the serverless handler (it starts ComfyUI itself)
cd "$V/worker"
log "starting handler"
exec python handler.py
