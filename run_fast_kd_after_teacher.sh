#!/usr/bin/env bash
set -euo pipefail

cd /home/jovyan/clipcap

TEACHER_TRAIN_PID="${TEACHER_TRAIN_PID:-793191}"
TEACHER_RUNNER_PID="${TEACHER_RUNNER_PID:-793190}"
TEACHER_WRAPPER_PID="${TEACHER_WRAPPER_PID:-793189}"
TEACHER_CKPT="${TEACHER_CKPT:-checkpoints/mscoco_teacher_retrain/mscoco_teacher_retrain-009.pt}"
OUT_DIR="${OUT_DIR:-checkpoints/mscoco_kd_fast_teacher_retrain}"

echo "[QUEUE] Waiting for teacher train process ${TEACHER_TRAIN_PID} to finish..."
while kill -0 "${TEACHER_TRAIN_PID}" 2>/dev/null; do
  date '+[QUEUE] %Y-%m-%d %H:%M:%S teacher is still training'
  sleep 60
done

if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "[ERROR] Teacher training process ended, but final checkpoint is missing: ${TEACHER_CKPT}" >&2
  exit 1
fi

echo "[QUEUE] Teacher checkpoint is ready: ${TEACHER_CKPT}"

if kill -0 "${TEACHER_RUNNER_PID}" 2>/dev/null; then
  echo "[QUEUE] Stopping teacher retrain runner to skip its slow full eval phase."
  pkill -TERM -P "${TEACHER_RUNNER_PID}" 2>/dev/null || true
  kill -TERM "${TEACHER_RUNNER_PID}" 2>/dev/null || true
  kill -TERM "${TEACHER_WRAPPER_PID}" 2>/dev/null || true
  sleep 10
  pkill -KILL -P "${TEACHER_RUNNER_PID}" 2>/dev/null || true
  kill -KILL "${TEACHER_RUNNER_PID}" 2>/dev/null || true
  kill -KILL "${TEACHER_WRAPPER_PID}" 2>/dev/null || true
fi

echo "[RUN] Fast KD fine-tuning from student CE"
PYTHONUNBUFFERED=1 python run_mscoco_kd_ablation.py \
  --modes student_logit_kd,student_prefix_kd,student_dual_kd \
  --teacher_checkpoint "${TEACHER_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --epochs 3 \
  --save_every 1 \
  --bs 64 \
  --lr 1e-5 \
  --warmup_steps 300 \
  --distill_temperature 4.0 \
  --logit_kd_weight 0.3 \
  --prefix_kd_weight 0.02 \
  --distill_prefix_loss cosine \
  --init_checkpoint checkpoints/mscoco_kd_ablation/student_ce/student_ce-009.pt \
  --skip_eval
