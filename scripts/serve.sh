#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# serve.sh — launch the optimized Qwen3.8-Flash-Next NVFP4 server on ONE DGX Spark (GB10), then hot-install
# the promoted optimization stack (scripts/install_stack.sh). Derived from the MiaAI-Lab single-Spark recipe.
#
# Requires (see README "Prerequisites"): the pinned image, the Mia NVFP4 checkpoint in the HF cache, the packed
# PLE table (scripts/prepare.sh), and the drafter patch built from adapters/ (scripts/build_adapter.sh).
#
# Environment (all optional):
#   Q38_NAME        container name            (default q38fn-opt)
#   Q38_PORT        port                      (default 5810)
#   Q38_IMAGE       image                     (default vllm/vllm-openai@sha256:fc120ece...)
#   Q38_HF_HOME     host HF cache             (default $HOME/.cache/huggingface)
#   Q38_PLE_CACHE   host packed-PLE dir root  (default $HOME/.cache/vllm/ple_cache)
#   Q38_STATE       logs + vLLM cache dir     (default $REPO/.state)
#   Q38_GMU / Q38_MAX_NUM_SEQS / Q38_MAX_MODEL_LEN   (defaults 0.786 / 4 / 262144)
#   Q38_NO_ADAPTER=1  serve the stock MTP drafter (skip the r32a patch)
#   Q38_NO_INSTALL=1  launch only; do not hot-install the optimization stack
#   Q38_BLOCK_DROP=1  stock trailing prefix-cache block drop (disable the E44 backport)
set -euo pipefail
. "$(dirname "$0")/common.sh"      # loads .env and shared defaults (MODEL_REVISION, IMAGE, HFH, PLEC, PORT, NAME, PATCH)
O=$REPO/files; RT=$REPO/overlays/runtime
PKG=/usr/local/lib/python3.12/dist-packages
TAG=${Q38_TAG:-opt}
EXP=${Q38_STATE:-$REPO/.state}
SPEC='{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
# E44: keep the trailing prefix-cache block (MiaAI-Lab PR #71, backport of vllm#53388; TTFT only, decode unchanged)
BD_MOUNTS=""
if [[ "${Q38_BLOCK_DROP:-0}" != 1 ]]; then
  BD=$REPO/overlays/block_drop/out
  [[ -f $BD/config/speculative.py ]] || { echo "missing $BD — run scripts/prepare.sh (or set Q38_BLOCK_DROP=1)" >&2; exit 2; }
  for f in $(cd "$BD" && find . -name '*.py' | sed 's|^./||'); do BD_MOUNTS="$BD_MOUNTS -v $BD/$f:$PKG/vllm/$f:ro"; done
  SPEC='{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"disable_eagle_block_drop":true}'
fi
COMP='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16]}'
MNS=${Q38_MAX_NUM_SEQS:-4}
GMU=${Q38_GMU:-0.786}
MML=${Q38_MAX_MODEL_LEN:-262144}
DV=$RT/dv_cur_u48k.txt
MTPF=$RT/mtp_exp_t8.py
if [[ "${Q38_NO_ADAPTER:-0}" != 1 ]]; then
  [[ -f $PATCH ]] || { echo "missing $PATCH — run scripts/build_adapter.sh first (or set Q38_NO_ADAPTER=1)" >&2; exit 2; }
  MTP_PATCH_ENV="-e EXP_MTP_PATCH=/q38/adapters/r32a/mtp_patch_r32a.safetensors"
else MTP_PATCH_ENV=""; fi
ls "$PLEC"/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/*.packed_u8 >/dev/null 2>&1 || { echo "packed PLE table missing — run scripts/prepare.sh" >&2; exit 2; }
if docker ps -a --format '{{.Names}}' | grep -q "^$NAME\$"; then echo "container $NAME already exists — run scripts/stop.sh first" >&2; exit 2; fi
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "port $PORT is in use (another server running?)" >&2; exit 2; fi
avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if (( avail < 104 )); then
  echo "REFUSING: only ${avail} GiB MemAvailable (need >=104 for a 100 GiB container + margin)" >&2; exit 2
fi

mkdir -p $EXP/logs $EXP/cache/vllm
# persistent JIT caches (Triton/CUDA), keyed by image digest; see docs/OPERATIONS.md "SSD hygiene"
IMGKEY=$(docker image inspect "$IMAGE" --format '{{.Id}}' | cut -c8-19)
JIT=$EXP/cache/jit/$IMGKEY; mkdir -p "$JIT/triton" "$JIT/nv"
LOG=$EXP/logs/${TAG}-$(date +%Y%m%dT%H%M%S)
echo "$TAG" > $EXP/logs/CURRENT_TAG

# shellcheck disable=SC2086
docker run -d --name $NAME \
  --log-driver json-file --log-opt max-size=50m --log-opt max-file=3 \
  -v $JIT/triton:/root/.triton -v $JIT/nv:/root/.nv \
  --label q38fn-opt=1 --label q38fn-opt.tag="$TAG" \
  --gpus all --network host --ipc host \
  --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
  --memory 100g --memory-swap 100g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_PACKED_TABLE_DIR=/root/.cache/vllm/ple_cache/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4 \
  -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300 \
  -e QWEN38_WEIGHT_LOAD_DIAGNOSTICS=1 -e QWEN38_BATCHED_EXPERT_LOAD=1 \
  -v $DV:/root/draft_vocab.txt:ro -e VLLM_MTP_DRAFT_VOCAB=/root/draft_vocab.txt \
  -v $O/chat-template/chat_template.jinja:/root/chat_template.jinja:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -v $RT/ple_layer_fixrows.py:$PKG/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -v $O/vllm/modelopt_patched.py:$PKG/vllm/model_executor/layers/quantization/modelopt.py:ro \
  -v $O/vllm/qsa_ops_patched.py:$PKG/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py:ro \
  -v $O/vllm/qsa_nvidia_patched.py:$PKG/vllm/models/qwen3_8_flash_next/nvidia/qsa.py:ro \
  -v $MTPF:$PKG/vllm/models/qwen3_8_flash_next/nvidia/mtp.py:ro \
  -v $O/vllm/model_patched.py:$PKG/vllm/models/qwen3_8_flash_next/nvidia/model.py:ro \
  -v $O/vllm/default_loader_patched.py:$PKG/vllm/model_executor/model_loader/default_loader.py:ro \
  -v $O/vllm/weight_utils_patched.py:$PKG/vllm/model_executor/model_loader/weight_utils.py:ro \
  -v $O/vllm/routed_experts_patched.py:$PKG/vllm/model_executor/layers/fused_moe/routed_experts.py:ro \
  -v $O/vllm/ple_offload/ple_offload_layer.py:$PKG/vllm/model_executor/layers/ple_offload_layer.py:ro \
  -v $O/vllm/ple_offload/connector.py:$PKG/vllm/v1/ple_offload/connector.py:ro \
  -v $RT/ple_worker_zc.py:$PKG/vllm/v1/ple_offload/worker.py:ro \
  -v $O/vllm/ple_offload/protocol.py:$PKG/vllm/v1/ple_offload/protocol.py:ro \
  -v $HFH:/root/.cache/huggingface:ro \
  -v $EXP/cache/vllm:/root/.cache/vllm \
  -v $PLEC:/root/.cache/vllm/ple_cache:ro \
  -v $RT:/exp:ro -v $REPO:/q38:ro -v $RT/exp_q38_ext.py:$PKG/exp_q38_ext.py:ro $BD_MOUNTS \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_SERVER_DEV_MODE=1 -e EXP_MTP_DRAFT_HEAD_FP8=1 $MTP_PATCH_ENV \
  "$IMAGE" \
  $MODEL_ID --revision $MODEL_REVISION --tokenizer-revision $MODEL_REVISION \
  --served-model-name ${Q38_SERVED_NAME:-qwen3.8-flash-next} --tensor-parallel-size 1 --gpu-memory-utilization $GMU \
  --max-num-seqs $MNS --max-num-batched-tokens 2048 --max-model-len $MML --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype bfloat16 --load-format safetensors --safetensors-load-strategy lazy \
  --enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice \
  --chat-template /root/chat_template.jinja --tool-call-parser qwen3_xml \
  --distributed-executor-backend mp \
  --speculative-config "$SPEC" \
  --compilation-config "$COMP" \
  --worker-extension-cls exp_q38_ext.ExpWorkerExt \
  --host 0.0.0.0 --port $PORT >/dev/null

# experiment-owned watchdog (frozen copy of production memwatch.sh, logs into the experiment dir)
MEMWATCH_LOG=$LOG-memwatch.log MEMWATCH_ARCHIVE_DIR=$EXP/logs/archive \
  nohup bash $REPO/scripts/memwatch.sh $NAME 6 > $LOG-memwatch.log 2>&1 &
echo "started $NAME tag=$TAG log=$LOG draft_vocab=$DV mtp=$MTPF"
t0=$(date +%s)
while true; do
  sleep 10
  if ! docker ps --format '{{.Names}}' | grep -q "^$NAME\$"; then
    docker logs --tail 400 $NAME > $LOG-container.log 2>&1 || true
    echo "CONTAINER EXITED after $(( $(date +%s)-t0 ))s; tail:"; tail -30 $LOG-container.log; exit 1
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health || true)
  if [[ "$code" == "200" ]]; then break; fi
  if (( $(date +%s)-t0 > 1500 )); then echo "TIMEOUT waiting for health"; exit 1; fi
done
docker logs $NAME > $LOG-container.log 2>&1 || true
echo "READY after $(( $(date +%s)-t0 ))s"
grep -E "Model loading took|Available KV|KV cache size|Graph capturing|MTP draft vocab|NvFp4 MoE backend|cudagraph_mode|init engine" $LOG-container.log | cut -c1-260 | tail -14

[[ "${Q38_NO_INSTALL:-0}" == 1 ]] && exit 0
Q38_PORT=$PORT bash "$REPO/scripts/install_stack.sh"
