#!/usr/bin/bash
#SBATCH -J AVDFD-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 2-0
#SBATCH -o logs/hyper_search-%A.out
#SBATCH --nodelist=ariel-v9

set -euo pipefail

# ==================== 경로 설정 ====================
PREPROC_ROOT="/local_datasets/jhlee39/FakeAVCeleb/preproc"
SPLIT_DIR="/data/jhlee39/workspace/repos/AVDFD/data/splits"

TRAIN_SPLIT="trainset_7_3.txt"
VAL_SPLIT="validset_7_3.txt"

TRAIN_FAKES="RVFA"
VAL_FAKES="RVFA,RVFA-VC"

# ==================== 기본 하이퍼파라미터 ====================
BATCH_SIZE=4
EPOCHS=80
LR=1e-5
NUM_WORKERS=6
WARMUP_EPOCHS=5
SEED=0
PATIENCE=10

AUDIO_PRETRAINED="audio_pretrained.pt"

# MARLIN / A2V 관련
BACKBONE_LR_SCALE=0.1     # vis_enc.backbone lr = LR * BACKBONE_LR_SCALE
A2V_NHEAD=4
A2V_LAYERS=2

# MARLIN backbone freeze 여부
# 기본은 freeze 켠 상태. backbone까지 finetune 하려면 "" 로 바꾸면 됨.
FREEZE_VIS_ENC="--freeze_vis_enc"   # or "" for finetuning backbone

# ==================== Grid search용 config (없으면 단일 실험) ====================
# TUNE_CONFIG=""  # 단일 실험 모드
TUNE_CONFIG="config/hyper_config.json"  # grid search 모드 사용 시 경로 지정

# ==================== 환경 변수 ====================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_DISABLE=1

export TMPDIR=/dev/shm/$USER
mkdir -p "$TMPDIR"
export TORCH_HOME=$TMPDIR/torch_home
export HF_HOME=$TMPDIR/hf_home

# ==================== GPU 개수 탐지 ====================
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC=${SLURM_GPUS_ON_NODE}
else
  NPROC=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
fi

echo "[Train] Dual-only, GPUs=$NPROC"

# ==================== 공통 추가 인자 설정 ====================
EXTRA_ARGS=(
  --preproc_root "$PREPROC_ROOT"
  --split_dir "$SPLIT_DIR"
  --train_split "$TRAIN_SPLIT"
  --val_split "$VAL_SPLIT"
  --train_fakes "$TRAIN_FAKES"
  --val_fakes "$VAL_FAKES"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --warmup_epochs "$WARMUP_EPOCHS"
  --num_workers "$NUM_WORKERS"
  --seed "$SEED"
  --audio_pretrained "$AUDIO_PRETRAINED"
  --patience "$PATIENCE"
  --backbone_lr_scale "$BACKBONE_LR_SCALE"
  --a2v_nhead "$A2V_NHEAD"
  --a2v_layers "$A2V_LAYERS"
)

# freeze_vis_enc 옵션 추가 (비어 있지 않을 때만)
if [[ -n "$FREEZE_VIS_ENC" ]]; then
  EXTRA_ARGS+=("$FREEZE_VIS_ENC")
fi

# grid search config가 있으면 --tune_config 추가
if [[ -n "$TUNE_CONFIG" ]]; then
  EXTRA_ARGS+=(--tune_config "$TUNE_CONFIG")
fi

# ==================== 실행 ====================
if [[ "$NPROC" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$NPROC" train.py "${EXTRA_ARGS[@]}"
else
  python train.py "${EXTRA_ARGS[@]}"
fi
