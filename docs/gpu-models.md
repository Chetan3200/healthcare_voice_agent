# Optional GPU model servers

The application connects to three independent servers. Its existing Python 3.11
`voice` extra and `uv.lock` stay unchanged; do **not** install any model-server
requirements into the application's `.venv`. Existing OpenAI and local-speech
launchers remain available.

| Stage | Provider | Server | Default port |
|---|---|---|---|
| STT | `nemotron` | Pinned NeMo-Speech.cpp, CUDA, local Nemotron Q8 GGUF | 8080 |
| TTS | `breeze` | Pinned Breeze Python streaming API | 7861 |
| LLM | `hybrid_diffusion` | Bundled modified SGLang/FlashInfer | 30000 |

The app's browser UI remains on **7860**, avoiding Breeze's upstream default
port conflict. `deploy/gpu-models.lock.json` records inspected source/model
revisions. This is **not** a full CUDA dependency or container lock. No real
GPU inference, memory fit, speech quality, tool correctness or latency has been
verified by the offline adapter tests.

## 1. Prepare the GPU host once

Use Linux with a compatible NVIDIA driver and CUDA toolkit. Check `nvidia-smi`,
`nvcc --version`, `cmake --version` (3.26+) and `uv --version`. HybridDiffusion's
upstream setup uses CUDA 12.8 wheels and Python **3.10**; the app uses **3.11**.
The upstream runtime was validated on H100, not every consumer GPU. Confirm
kernel/toolkit compatibility on the actual allocated GPU.

The following are explicit **future installation/download commands**, not
launcher side effects. Run them yourself on the GPU host. Set paths outside the
application checkout to avoid accidental commits and dependency contamination:

```bash
export GPU_ROOT="$HOME/healthcare-voice-models"
mkdir -p "$GPU_ROOT/src" "$GPU_ROOT/models" "$GPU_ROOT/envs"
export NEMO_SOURCE_DIR="$GPU_ROOT/src/NeMo-Speech.cpp"
export BREEZE_SOURCE_DIR="$GPU_ROOT/src/breeze-tts"
export HYBRID_SOURCE_DIR="$GPU_ROOT/src/HybridDiffusion"
export BREEZE_PYTHON="$GPU_ROOT/envs/breeze/bin/python"
export HYBRID_DIFFUSION_CACHE_ROOT="$GPU_ROOT/hybrid-cache"
export NEMO_MODEL_PATH="$GPU_ROOT/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
export BREEZE_MODEL_PATH="$GPU_ROOT/models/Breeze-TTS-2"
export HYBRID_MODEL_PATH="$GPU_ROOT/models/HybridDiffusion-2B"
```

Keep these exports in a private shell file outside Git if useful. Do not copy
`.env.example` over an existing `.env`, and do not source the application's
`.env` into third-party model-server processes.

### Nemotron native server

Install build prerequisites through your host's approved package manager:
C/C++ toolchain, Git, pkg-config, CMake 3.26+, Ninja, SentencePiece development
headers, and a compatible CUDA 12/13 toolkit. On Ubuntu the basic packages are
`build-essential cmake ninja-build git pkg-config libsentencepiece-dev`.

```bash
git clone https://github.com/NVIDIA/NeMo-Speech.cpp.git "$NEMO_SOURCE_DIR"
git -C "$NEMO_SOURCE_DIR" checkout --detach 07003daa7eefea542076310722ccaa89709ee3c3
git -C "$NEMO_SOURCE_DIR" submodule update --init ggml third_party/cpp-httplib llama.cpp
(
  cd "$NEMO_SOURCE_DIR"
  scripts/configure.sh cuda-server
  cmake --build --preset cuda-server
)
export NEMO_BIN="$NEMO_SOURCE_DIR/build/cuda-server/bin/nemo-speech"
"$NEMO_BIN" doctor
# Explicit download. -f rejects HTTP errors; the digest is checked below and at launch.
curl --fail --location --proto '=https' --proto-redir '=https' \
  'https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/1c8deaecc64b91f034d73e08dd8b64625eb3395d/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf' \
  --output "$NEMO_MODEL_PATH"
printf '%s  %s\n' \
  a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae \
  "$NEMO_MODEL_PATH" | sha256sum --check
```

Use `scripts/configure.sh`, not bare CMake: it applies the pinned GGML patches.
The launcher passes an existing GGUF path, disables other capabilities, and
rejects inherited `NEMO_SPEECH_*` variables so an unrelated environment setting
cannot silently enable/download a companion model. Local Pipecat VAD/Smart Turn,
not server endpointing, owns turn boundaries. `doctor` must show the intended
CUDA backend; a source checkout pin alone does not prove a binary was rebuilt.

### Breeze isolated environment

```bash
git clone https://github.com/breezeblue-ai/breeze-tts.git "$BREEZE_SOURCE_DIR"
git -C "$BREEZE_SOURCE_DIR" checkout --detach 008f769016b0a24711becd7a4925030bc93f608c
uv venv --python 3.10 "$GPU_ROOT/envs/breeze"
uv pip install --python "$BREEZE_PYTHON" \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchaudio==2.9.1
uv pip install --python "$BREEZE_PYTHON" -r "$BREEZE_SOURCE_DIR/requirements.txt"
# Native compilation needs the CUDA toolkit, C++ toolchain and build helpers.
uv pip install --python "$BREEZE_PYTHON" setuptools wheel packaging ninja
# Example target: H100/Hopper = 90. Use 80 for A100, or the actual target GPU.
FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=8 uv pip install --python "$BREEZE_PYTHON" \
  --no-build-isolation --no-deps flash-attn==2.8.3
"$(dirname "$BREEZE_PYTHON")/hf" download BreezeBlue/Breeze-TTS-2 \
  --revision 3e28c5151381a722f1d8661b4118c298caa77aa4 --local-dir "$BREEZE_MODEL_PATH"
uv pip check --python "$BREEZE_PYTHON"
uv pip freeze --python "$BREEZE_PYTHON" > "$GPU_ROOT/breeze-installed.freeze.txt"
```

These versions follow the pinned upstream requirements and Docker recipe; some
transitive dependencies remain unpinned. Keep the resulting freeze and actual
GPU/driver information after validating. Do not substitute the app environment.
The app currently uses **voice design**, a style instruction with no reference
audio. It does not expose voice-cloning file upload. Breeze lists English and
Chinese; Hindi/Hinglish pronunciation is not established.

### HybridDiffusion isolated environment

```bash
git clone https://github.com/yuchen-zhu-zyc/HybridDiffusion.git "$HYBRID_SOURCE_DIR"
git -C "$HYBRID_SOURCE_DIR" checkout --detach 6ca547aebb72bfe897e80e0a683c776789e5f38c
# Ensure python3.10 is installed, or set PYTHON_BIN to that interpreter explicitly.
PYTHON_BIN="$(command -v python3.10)" \
  bash "$HYBRID_SOURCE_DIR/eval/scripts/setup_eval_env.sh"
export EVAL_VENV="$HYBRID_DIFFUSION_CACHE_ROOT/venvs/hybrid-diffusion-eval"
"$EVAL_VENV/bin/hf" download yuchen-zhu-zyc/HybridDiffusion-2B \
  --revision 04e16a066c17a512f9a567b01362d6ab811bcf4b --local-dir "$HYBRID_MODEL_PATH"
```

Use the **bundled** SGLang and modified FlashInfer together, not `pip install
sglang` in the app. The upstream setup creates its own environment, native build
data and freeze manifest. Do not pass radix-off/no-buffer options for `self-spec`.
The launcher selects `self-spec` by default; set `GPU_HYBRID_MODE=diffusion` or
`causal` for explicit alternatives. SSE chunks may contain several tokens.
`qwen3_coder` is the checkpoint's tool parser; plain `qwen` is the wrong format.
The app disables thinking through `chat_template_kwargs.enable_thinking=false`.
The clinical tool backends themselves remain outside this integration's scope.

## 2. Start the prepared servers

From the **application repository**, run one command per terminal, with the
exports above present in each terminal:

```bash
./scripts/serve_gpu_model.sh nemotron
./scripts/serve_gpu_model.sh breeze
./scripts/serve_gpu_model.sh hybrid_diffusion
```

These scripts perform no installation/download and fail for missing local assets
or wrong source revisions. They bind only `127.0.0.1`; startup uses offline model
hub flags. Server-side JIT compilation can still occur and may take time.

Defaults: Breeze eager mode; Hybrid concurrency 1, CUDA graph batch size 1 and
`GPU_HYBRID_MEM_FRACTION=0.40`. These are conservative starting settings, **not a
promise that three models fit or run quickly on one GPU**. If loading fails,
measure memory per server, change the fraction, or use separate GPUs/process
`CUDA_VISIBLE_DEVICES` assignments. Opt into Breeze's larger-memory fast path
only after checking capacity: `GPU_BREEZE_FAST=true`.

## 3. Connect the app

On the laptop or app host, install only the existing app environment:

```bash
uv sync --locked --extra voice
./scripts/voice_gpu.sh --check-config
```

For a remote GPU host, leave all model servers bound to loopback and open an SSH
tunnel from the app host. Replace `your-gpu-host` with your SSH target:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8080:127.0.0.1:8080 \
  -L 127.0.0.1:7861:127.0.0.1:7861 \
  -L 127.0.0.1:30000:127.0.0.1:30000 your-gpu-host
```

Wait for all servers to report ready, then:

```bash
LIVE_API_ENABLED=true ./scripts/voice_gpu.sh
```

Open `http://127.0.0.1:7860/client/`. Start with synthetic English speech. The
fully self-hosted selection needs no OpenAI key, but the explicit live-service
opt-in remains. Mixed provider selections that retain an OpenAI stage still
require its key and may incur API charges. Never expose these unauthenticated
model ports publicly or put credentials in URLs.

## Validation and rollback

Offline tests exercise protocol/configuration/lifecycle behavior, not GPU
kernels or actual speech. On the allocated GPU, check each server alone before
all three: load errors, first and warmed requests, Hindi/English ASR examples,
first audio, interruption, reconnect, long pauses, simultaneous memory use,
and empty/failing requests. Compare server traces with audible output; do not
interpret server first-byte timing as measured browser playback latency.

Two upstream protocol limits remain important during that check:

- Nemotron's loaded server checkpoint selects the model, not the client's model
  label. The launcher verifies its GGUF. Partial replacement hypotheses are
  withheld; complete text is released after the ordered commit acknowledgement.
- Breeze can keep GPU work running after a client interruption. An immediate
  subsequent request may receive HTTP 409; the adapter fails safely rather than
  retrying or replaying stale speech. Client cancellation/drain is tested, but
  does not prove server cancellation. Its PCM stream has no semantic completion
  marker, so a clean even-length EOF cannot prove every word was synthesized.

Stop the model servers/tunnel with Ctrl-C. Start the existing local-speech or
OpenAI path to roll back; neither the default configuration nor private `.env`
needs replacing. This guide does not initialize Git, commit, or push anything.

### Inspected source references

- [NeMo model artifact/digest index](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/07003daa7eefea542076310722ccaa89709ee3c3/models/index.json)
- [NeMo build presets](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/07003daa7eefea542076310722ccaa89709ee3c3/docs/build.md), [server flags](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/07003daa7eefea542076310722ccaa89709ee3c3/app/serve.cpp)
- [Breeze API CLI](https://github.com/breezeblue-ai/breeze-tts/blob/008f769016b0a24711becd7a4925030bc93f608c/breeze_infer/api.py), [requirements](https://github.com/breezeblue-ai/breeze-tts/blob/008f769016b0a24711becd7a4925030bc93f608c/requirements.txt), [CUDA recipe](https://github.com/breezeblue-ai/breeze-tts/blob/008f769016b0a24711becd7a4925030bc93f608c/docker/Dockerfile)
- [HybridDiffusion setup](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/6ca547aebb72bfe897e80e0a683c776789e5f38c/eval/scripts/setup_eval_env.sh), [serving](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/6ca547aebb72bfe897e80e0a683c776789e5f38c/eval/scripts/serve.sh)

### Offline implementation checkpoint

- **587 offline tests passed**, including native Pipecat construction, fake
  WebSocket/HTTP/SSE protocol tests, cancellation/late-result guards, streaming
  PCM validation, native context keepalives, and stubbed launcher contracts.
- `uv lock --check --offline --no-python-downloads`, Python compilation, both
  launcher syntax checks and `voice_gpu.sh --check-config` passed.
- Fingerprinted private `.env`, dependency files, Compose, fixtures, migrations,
  and existing local STT/TTS/splitting implementations were unchanged.
- No dependencies were installed, model weights downloaded, or native GPU
  inference executed in this implementation pass. Only protocol/mock backends
  were exercised; real GPU/browser validation remains outstanding.
