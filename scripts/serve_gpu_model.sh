#!/usr/bin/env bash
# Start an explicitly prepared model server. No installs, clones or downloads.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
die() { printf 'GPU server: %s\n' "$*" >&2; exit 1; }
usage() {
  printf '%s\n' 'Usage: scripts/serve_gpu_model.sh nemotron|breeze|hybrid_diffusion|qwen' \
    'Prepare separate runtimes and local weights first: docs/gpu-models.md'
}
[[ $# == 1 ]] || { usage; exit 2; }
[[ "$1" != --help && "$1" != -h ]] || { usage; exit 0; }
require_dir() { [[ -d "$1" ]] || die "$2 directory is missing; see docs/gpu-models.md"; }
require_file() { [[ -s "$1" ]] || die "$2 file is missing or empty: $1; see docs/gpu-models.md"; }
require_exec() { [[ -x "$1" ]] || die "$2 executable is missing; see docs/gpu-models.md"; }
require_checkout() {
  require_dir "$1" "$3"
  [[ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" == "$2" ]] || die "$3 must use the revision in deploy/gpu-models.lock.json"
}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
case "$1" in
  nemotron)
    : "${NEMO_SOURCE_DIR:?Set NEMO_SOURCE_DIR to the prepared pinned checkout}"
    : "${NEMO_MODEL_PATH:?Set NEMO_MODEL_PATH to the pinned local Q8 GGUF}"
    require_checkout "$NEMO_SOURCE_DIR" 07003daa7eefea542076310722ccaa89709ee3c3 NeMo-Speech.cpp
    NEMO_BIN="${NEMO_BIN:-$NEMO_SOURCE_DIR/build/cuda-server/bin/nemo-speech}"
    require_exec "$NEMO_BIN" nemo-speech
    require_file "$NEMO_MODEL_PATH" Nemotron
    # Native NeMo has its own downloader, independent of HF_HUB_OFFLINE. Reject
    # inherited engine settings that might activate/download companion models.
    if env | grep '^NEMO_SPEECH_' >/dev/null; then
      die 'Unset NEMO_SPEECH_* settings before this ASR-only launcher'
    fi
    if command -v sha256sum >/dev/null; then
      digest="$(sha256sum "$NEMO_MODEL_PATH" | awk '{print $1}')"
    else
      digest="$(shasum -a 256 "$NEMO_MODEL_PATH" | awk '{print $1}')"
    fi
    [[ "$digest" == a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae ]] || die 'Nemotron GGUF checksum does not match the pinned artifact'
    exec "$NEMO_BIN" serve --host 127.0.0.1 --port 8080 --no-ui \
      --asr-model "$NEMO_MODEL_PATH" --backend cuda \
      --asr.endpointing.enable=false --asr.batching.enabled=false \
      --nmt.enabled=false --tts.enabled=false
    ;;
  breeze)
    : "${BREEZE_SOURCE_DIR:?Set BREEZE_SOURCE_DIR to the prepared pinned checkout}"
    : "${BREEZE_PYTHON:?Set BREEZE_PYTHON to the isolated Breeze environment interpreter}"
    : "${BREEZE_MODEL_PATH:?Set BREEZE_MODEL_PATH to the downloaded checkpoint directory}"
    require_checkout "$BREEZE_SOURCE_DIR" 008f769016b0a24711becd7a4925030bc93f608c Breeze
    require_exec "$BREEZE_PYTHON" Breeze-Python
    require_dir "$BREEZE_MODEL_PATH" Breeze-model
    for file in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json \
      model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
      audio_tokenizer/config.json audio_tokenizer/model.safetensors; do
      require_file "$BREEZE_MODEL_PATH/$file" Breeze-checkpoint
    done
    # Convert paths before changing working directory; module imports must come
    # from this pinned source tree, not the voice application's environment.
    BREEZE_PYTHON="$(cd "$(dirname "$BREEZE_PYTHON")" && pwd)/$(basename "$BREEZE_PYTHON")"
    BREEZE_MODEL_PATH="$(cd "$BREEZE_MODEL_PATH" && pwd)"
    [[ "$BREEZE_PYTHON" != "$ROOT/.venv/"* ]] || die 'Breeze must not use the application .venv'
    cd "$BREEZE_SOURCE_DIR"
    set --
    case "${GPU_BREEZE_FAST:-false}" in
      false) ;;
      true) set -- --fast-all ;;
      *) die 'GPU_BREEZE_FAST must be true or false' ;;
    esac
    exec "$BREEZE_PYTHON" -m breeze_infer.api "$BREEZE_MODEL_PATH" \
      --host 127.0.0.1 --port 7861 "$@"
    ;;
  hybrid_diffusion)
    : "${HYBRID_SOURCE_DIR:?Set HYBRID_SOURCE_DIR to the prepared pinned checkout}"
    : "${HYBRID_DIFFUSION_CACHE_ROOT:?Set HYBRID_DIFFUSION_CACHE_ROOT to its prepared cache}"
    : "${HYBRID_MODEL_PATH:?Set HYBRID_MODEL_PATH to the downloaded checkpoint directory}"
    require_checkout "$HYBRID_SOURCE_DIR" 6ca547aebb72bfe897e80e0a683c776789e5f38c HybridDiffusion
    require_dir "$HYBRID_MODEL_PATH" HybridDiffusion-model
    for file in config.json tokenizer_config.json tokenizer.json \
      model.safetensors.index.json model-00001-of-00001.safetensors; do
      require_file "$HYBRID_MODEL_PATH/$file" HybridDiffusion-checkpoint
    done
    mode="${GPU_HYBRID_MODE:-self-spec}"
    case "$mode" in self-spec|diffusion|causal) ;; *) die 'GPU_HYBRID_MODE must be self-spec, diffusion, or causal' ;; esac
    export EVAL_VENV="${EVAL_VENV:-$HYBRID_DIFFUSION_CACHE_ROOT/venvs/hybrid-diffusion-eval}"
    require_exec "$EVAL_VENV/bin/python" HybridDiffusion-Python
    [[ "$EVAL_VENV" != "$ROOT/.venv" ]] || die 'HybridDiffusion must not use the application .venv'
    # Explicitly override upstream's public bind and 80% memory reservation.
    # This is a planning default, NOT a guarantee all three models fit one GPU.
    export HOST=127.0.0.1 PORT=30000 MAX_RUNNING_REQUESTS=1
    export MEM_FRACTION_STATIC="${GPU_HYBRID_MEM_FRACTION:-0.40}"
    [[ "$MEM_FRACTION_STATIC" =~ ^0\.[0-9]+$ ]] || die 'GPU_HYBRID_MEM_FRACTION must be between 0 and 1'
    [[ "$MEM_FRACTION_STATIC" =~ [1-9] ]] || die 'GPU_HYBRID_MEM_FRACTION must be positive'
    export CUDA_GRAPH_BS=1
    exec bash "$HYBRID_SOURCE_DIR/eval/scripts/serve.sh" "$mode" "$HYBRID_MODEL_PATH" -- \
      --served-model-name yuchen-zhu-zyc/HybridDiffusion-2B \
      --tool-call-parser qwen3_coder
    ;;
  qwen)
    : "${QWEN_PYTHON:?Set QWEN_PYTHON to the separate vLLM environment interpreter}"
    : "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the local Qwen3.8-27B-FP8 checkpoint}"
    : "${CUDA_VISIBLE_DEVICES:?Select two free GPUs explicitly, e.g. CUDA_VISIBLE_DEVICES=2,3}"
    require_exec "$QWEN_PYTHON" Qwen-Python
    require_dir "$QWEN_MODEL_PATH" Qwen-model
    QWEN_PYTHON="$(cd "$(dirname "$QWEN_PYTHON")" && pwd)/$(basename "$QWEN_PYTHON")"
    QWEN_MODEL_PATH="$(cd "$QWEN_MODEL_PATH" && pwd)"
    [[ "$QWEN_PYTHON" != "$ROOT/.venv/"* ]] || die 'Qwen must not use the application .venv'
    [[ "$("$QWEN_PYTHON" -c 'from importlib.metadata import version; print(version("vllm"))')" == 0.30.0 ]] || die 'Qwen requires vLLM 0.30.0 in its separate environment'
    for file in config.json tokenizer_config.json tokenizer.json chat_template.jinja \
      model.safetensors.index.json outside.safetensors mtp.safetensors; do
      require_file "$QWEN_MODEL_PATH/$file" Qwen-checkpoint
    done
    for layer in {0..63}; do
      require_file "$QWEN_MODEL_PATH/layers-$layer.safetensors" Qwen-checkpoint
    done
    # Reuse the existing LLM port/forward. Stop HybridDiffusion manually first.
    # No speculative decoding or extra serving layer for this initial trial.
    exec "$QWEN_PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$QWEN_MODEL_PATH" --served-model-name Qwen/Qwen3.8-27B-FP8 \
      --host 127.0.0.1 --port 30000 --tensor-parallel-size 2 \
      --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.90 \
      --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking":false}'
    ;;
  *) usage; exit 2 ;;
esac
