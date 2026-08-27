#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the 70k local LIA-X motion-tokenizer checkpoint with the same
# protocol used for the previous 30k/40k/50k/60k comparison:
#   HDTF first 100 records, GT-motion reconstruction, full-frame FID/FVD,
#   paired Sync-C/Sync-D, and BF16 decoder-only RTF.
#
# Common overrides:
#   EVAL_GPU=4 SYNC_GPUS=4,5 LIMIT=100 bash tools/eval_liax_70k_all_metrics.sh
#   RUN_RECON=0 bash tools/eval_liax_70k_all_metrics.sh   # reuse videos
#   CHECKPOINT=/path/to/model.pt OUTPUT_DIR=/path/to/out bash ...

ckp=80000

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIBE_PYTHON="${VIBE_PYTHON:-/data/joe/anaconda3/envs/vibe/bin/python}"
SYNC_PYTHON="${SYNC_PYTHON:-/data/joe/anaconda3/envs/syncnet/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-/usr/bin/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-/usr/bin/ffprobe}"

CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/exp/motion_tokenizer_liax_baseline_stable/step_0000${ckp}.pt}"
MANIFEST="${MANIFEST:-/nfs-speech-cfs/wangzhou/s2s/vibehead/data/hdtf/test.jsonl}"
GT_DIR="${GT_DIR:-/nfs-speech-cfs/wangzhou/s2s/vibehead/data/hdtf/clips}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/liax_baseline_stable_${ckp}_hdtf100}"
REFERENCE_VIDEO="${REFERENCE_VIDEO:-${GT_DIR}/RD_Radio14_000_729_809.mp4}"
SYNC_CAL="${SYNC_CAL:-/nfs-speech-cfs/wangzhou/s2s/data/syncnet_python/cal.py}"

EVAL_GPU="${EVAL_GPU:-4}"
SYNC_GPUS="${SYNC_GPUS:-${EVAL_GPU}}"
LIMIT="${LIMIT:-100}"
RESOLUTION="${RESOLUTION:-512}"
ENCODE_BATCH="${ENCODE_BATCH:-32}"
RENDER_CHUNK="${RENDER_CHUNK:-4}"
H264_CRF="${H264_CRF:-18}"
H264_PRESET="${H264_PRESET:-medium}"
RTF_BATCH_SIZES="${RTF_BATCH_SIZES:-1,4,8}"
RTF_WARMUP="${RTF_WARMUP:-10}"
RTF_REPEATS="${RTF_REPEATS:-30}"

RUN_RECON="${RUN_RECON:-1}"
RUN_FID_FVD="${RUN_FID_FVD:-1}"
RUN_SYNC="${RUN_SYNC:-1}"
RUN_RTF="${RUN_RTF:-1}"

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/metrics_logs"
mkdir -p "${LOG_DIR}"

for required_file in \
    "${CHECKPOINT}" \
    "${MANIFEST}" \
    "${REFERENCE_VIDEO}" \
    "${SYNC_CAL}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -x "${VIBE_PYTHON}" ]]; then
    echo "Vibe Python is not executable: ${VIBE_PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${SYNC_PYTHON}" ]]; then
    echo "SyncNet Python is not executable: ${SYNC_PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${FFMPEG_BIN}" ]]; then
    echo "ffmpeg is not executable: ${FFMPEG_BIN}" >&2
    exit 1
fi
if [[ ! -x "${FFPROBE_BIN}" ]]; then
    echo "ffprobe is not executable: ${FFPROBE_BIN}" >&2
    exit 1
fi
if ! "${FFMPEG_BIN}" -hide_banner -encoders 2>/dev/null | grep 'libx264' >/dev/null; then
    echo "ffmpeg has no libx264 encoder: ${FFMPEG_BIN}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

echo "checkpoint=${CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"
echo "limit=${LIMIT} eval_gpu=${EVAL_GPU} sync_gpus=${SYNC_GPUS}"

if [[ "${RUN_RECON}" == "1" ]]; then
    echo "[1/4] Reconstructing GT-motion videos..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
        "${VIBE_PYTHON}" -m twinlakes.bin.reconstruct_motion_tokenizer \
        --checkpoint "${CHECKPOINT}" \
        --manifest "${MANIFEST}" \
        --output_dir "${OUTPUT_DIR}" \
        --limit "${LIMIT}" \
        --resolution "${RESOLUTION}" \
        --encode_batch "${ENCODE_BATCH}" \
        --render_chunk "${RENDER_CHUNK}" \
        --video_codec h264 \
        --h264_crf "${H264_CRF}" \
        --h264_preset "${H264_PRESET}" \
        --ffmpeg_bin "${FFMPEG_BIN}" \
        --mux_audio \
        2>&1 | tee "${LOG_DIR}/reconstruct.log"
else
    echo "[1/4] Reconstruction skipped (RUN_RECON=${RUN_RECON})."
fi

VIDEO_COUNT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' ! -name '*.noaudio.mp4' | wc -l)"
if (( VIDEO_COUNT < LIMIT )); then
    echo "Only ${VIDEO_COUNT}/${LIMIT} reconstructed videos found in ${OUTPUT_DIR}." >&2
    exit 1
fi
echo "Reconstructed videos available: ${VIDEO_COUNT}"

NON_H264_COUNT=0
while IFS= read -r -d '' video_path; do
    codec_name="$(
        "${FFPROBE_BIN}" -v error -select_streams v:0 \
            -show_entries stream=codec_name -of default=nw=1:nk=1 \
            "${video_path}"
    )"
    if [[ "${codec_name}" != "h264" ]]; then
        echo "Non-H.264 output: ${video_path} (codec=${codec_name})" >&2
        NON_H264_COUNT=$((NON_H264_COUNT + 1))
    fi
done < <(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' \
    ! -name '*.noaudio.mp4' -print0)
if (( NON_H264_COUNT > 0 )); then
    echo "Found ${NON_H264_COUNT} non-H.264 videos; refusing mixed-codec evaluation." >&2
    exit 1
fi
echo "Video codec check passed: all ${VIDEO_COUNT} outputs are H.264."

if [[ "${RUN_FID_FVD}" == "1" ]]; then
    echo "[2/4] Computing full-frame FID/FVD..."
    FIDFVD_WORK="/dev/shm/fidfvd_liax70k_${USER:-user}" \
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
        "${VIBE_PYTHON}" tools/eval_fid_fvd_robust.py \
        --gen_dir "${OUTPUT_DIR}" \
        --gt_dir "${GT_DIR}" \
        --limit "${LIMIT}" \
        --fid_side 256 \
        --fvd_frames 16 \
        2>&1 | tee "${LOG_DIR}/fid_fvd.log"
else
    echo "[2/4] FID/FVD skipped (RUN_FID_FVD=${RUN_FID_FVD})."
fi

if [[ "${RUN_SYNC}" == "1" ]]; then
    echo "[3/4] Computing paired Sync-C/Sync-D..."
    "${SYNC_PYTHON}" "${SYNC_CAL}" \
        --gen_dir "${OUTPUT_DIR}" \
        --gt_dir "${GT_DIR}" \
        --limit "${LIMIT}" \
        --gpus "${SYNC_GPUS}" \
        2>&1 | tee "${LOG_DIR}/syncnet.log"
else
    echo "[3/4] SyncNet skipped (RUN_SYNC=${RUN_SYNC})."
fi

if [[ "${RUN_RTF}" == "1" ]]; then
    echo "[4/4] Benchmarking BF16 decoder-only RTF..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
        "${VIBE_PYTHON}" tools/benchmark_liax_decoder_rtf.py \
        --kind local \
        --checkpoint "${CHECKPOINT}" \
        --image "${REFERENCE_VIDEO}" \
        --batch_sizes "${RTF_BATCH_SIZES}" \
        --fps 25 \
        --warmup "${RTF_WARMUP}" \
        --repeats "${RTF_REPEATS}" \
        --dtype bf16 \
        2>&1 | tee "${LOG_DIR}/decoder_rtf.log"
else
    echo "[4/4] RTF benchmark skipped (RUN_RTF=${RUN_RTF})."
fi

SYNC_REPORT="${OUTPUT_DIR}/syncnet_report_$(basename "${OUTPUT_DIR}").txt"
echo "Evaluation finished. Logs: ${LOG_DIR}"
if [[ -f "${LOG_DIR}/fid_fvd.log" ]]; then
    grep -E '配对 clip 数|FID \(|FVD \(' "${LOG_DIR}/fid_fvd.log" || true
fi
if [[ -f "${SYNC_REPORT}" ]]; then
    awk '/^\[生成 \(ours\)\]/{print; getline; print; getline; print}' \
        "${SYNC_REPORT}" || true
    echo "SyncNet report: ${SYNC_REPORT}"
fi
if [[ -f "${LOG_DIR}/decoder_rtf.log" ]]; then
    grep -E '^kind=|^batch=' "${LOG_DIR}/decoder_rtf.log" || true
fi
