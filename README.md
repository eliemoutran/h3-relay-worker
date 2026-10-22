# h3-relay-worker

RunPod Serverless worker for MiniMax H3 character swap using
[akatz-ai/h3-relay](https://github.com/akatz-ai/h3-relay) 2.0 and the
[H3 Character Swap LoRA v1.5](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA),
reproducing the author's qualified runtime: ComfyUI `79be670e`, frontend 1.53.6,
PyTorch 2.11 + CUDA 12.8, SAM 3.1 masking, 8 steps, CFG 1, `er_sde`, normal scheduler.

No custom image. The endpoint runs the stock `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
image with `bootstrap.sh` as its command. The bootstrap keeps a Python venv, ComfyUI, the
custom nodes, a static ffmpeg and ~44 GB of models on the attached network volume under
`/runpod-volume/relay`, so only the first boot per volume pays the install (~10 min).

Files

- `bootstrap.sh` – cold-start installer / launcher
- `handler.py` – RunPod handler: inputs in, SAM-masked swap, mp4 out (inline or chunked fetch)
- `swap_prompt.json` – the author's `H3-Character-Swap-Relay-V2.api.json` with our file slots
- `swap_sls.py` – client: submit, wait, download, write a JSON report with the billed cost

Endpoint command

```
/bin/bash -c "curl -sL https://raw.githubusercontent.com/eliemoutran/h3-relay-worker/main/bootstrap.sh | bash"
```

Model weights are downloaded from their upstream repositories at first boot and are
licensed by their authors; this repository redistributes none of them.
