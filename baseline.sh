#!/usr/bin/bash
#SBATCH -J AVDFD-baseline
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 2-0
#SBATCH -o logs/baseline-%A.out
#SBATCH --nodelist=ariel-v9

set -euo pipefail

# =========================================
# User settings
# =========================================

PREPROC_ROOT="/local_datasets/jhlee39/FakeAVCeleb/preproc"
SPLIT_DIR="/data/jhlee39/workspace/repos/AVDFD/data/splits"

TRAIN_SPLIT="trainset_7_3.txt" # trainset_7_3.txt trainset_70.txt
VAL_SPLIT="validset_7_3.txt" # validset_7_3.txt validset_70.txt

REAL_VERSION="RVRA"
TRAIN_FAKES="RVFA"
VAL_FAKES="RVFA,RVFA-VC"

# audio source: mel_err or mel
AUDIO_SUBDIR="mel"

BATCH_SIZE=16
EPOCHS=50
LR=1e-5
NUM_WORKERS=6
TARGET_LENGTH=1024
FREQM=0
TIMEM=0
NOISE=false

LOSS="CE"     # or BCE
METRICS="mAP" # or acc

HEAD_LR=50
LRSCHED_START=2
LRSCHED_STEP=1
LRSCHED_DECAY=0.5

PRETRAIN_PATH="OpenAVFF/checkpoints/stage3_init_from_stage2.pth"  # 필요 시 경로 조정

# =========================================
# Environment
# =========================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_DISABLE=1

export TMPDIR=/dev/shm/$USER
mkdir -p "$TMPDIR"
mkdir -p logs

export TORCH_HOME=$TMPDIR/torch_home
export HF_HOME=$TMPDIR/hf_home

# =========================================
# Run
# =========================================

echo "[Baseline] PREPROC_ROOT = $PREPROC_ROOT"
echo "[Baseline] AUDIO_SUBDIR  = $AUDIO_SUBDIR"

python baseline.py \
  --preproc_root "$PREPROC_ROOT" \
  --split_dir "$SPLIT_DIR" \
  --train_split "$TRAIN_SPLIT" \
  --val_split "$VAL_SPLIT" \
  --real_version "$REAL_VERSION" \
  --train_fakes "$TRAIN_FAKES" \
  --val_fakes "$VAL_FAKES" \
  --audio_subdir "$AUDIO_SUBDIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --n_epochs "$EPOCHS" \
  --target_length "$TARGET_LENGTH" \
  --freqm "$FREQM" \
  --timem "$TIMEM" \
  --noise "$NOISE" \
  --loss "$LOSS" \
  --metrics "$METRICS" \
  --head_lr "$HEAD_LR" \
  --lrscheduler_start "$LRSCHED_START" \
  --lrscheduler_step "$LRSCHED_STEP" \
  --lrscheduler_decay "$LRSCHED_DECAY" \
  --pretrain_path "$PRETRAIN_PATH"
