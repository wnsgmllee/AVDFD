#!/usr/bin/bash
#SBATCH -J AVDFD-pretrain
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 2-0
#SBATCH -o logs/pretrain-%A.out
#SBATCH --nodelist=ariel-v9

set -euo pipefail

# =========================================
# User settings
# =========================================

PREPROC_ROOT="/local_datasets/jhlee39/FakeAVCeleb/preproc"
SPLIT_DIR="/data/jhlee39/workspace/repos/AVDFD/data/splits"

TRAIN_SPLIT="trainset_7_3.txt" # trainset_7_3.txt trainset_70.txt
VAL_SPLIT="validset_7_3.txt" # validset_7_3.txt validset_70.txt

TRAIN_FAKES="RVFA" # RVFA RVFA-VC RVFA-SVC
VAL_FAKES="RVFA,RVFA-VC"

BATCH_SIZE=32
EPOCHS=80
LR=3e-5
NUM_WORKERS=6
WARMUP_EPOCHS=5
SEED=0

SAVE_PATH="pt_results/audio_pretrained_temp.pt"

# =========================================
# dire_mode: true  → mel_err (DIRE)
#           false → mel (raw mel spectrogram)
# =========================================
DIRE_MODE=false   # <- 여기서 true / false 설정

# =========================================
# 모델 파라미터 파일 저장 여부
# true  → 학습 후 best audio_enc를 SAVE_PATH에 저장
# false → 학습만 하고 저장 안 함
# =========================================
SAVE_MODEL=false  # <- 여기서 true / false 설정

# =========================================
# Environment
# =========================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_DISABLE=1

export TMPDIR=/dev/shm/$USER
mkdir -p "$TMPDIR"
mkdir -p pt_results
mkdir -p logs

export TORCH_HOME="$TMPDIR/torch_home"
export HF_HOME="$TMPDIR/hf_home"

# =========================================
# Run
# =========================================

echo "[Pretrain] DIRE_MODE  = $DIRE_MODE"
echo "[Pretrain] SAVE_MODEL = $SAVE_MODEL"

EXTRA_ARGS=""

if [ "$DIRE_MODE" = true ]; then
  EXTRA_ARGS="$EXTRA_ARGS --dire_mode"
fi

if [ "$SAVE_MODEL" = false ]; then
  EXTRA_ARGS="$EXTRA_ARGS --no_save"
fi

python pretrain.py \
  --preproc_root "$PREPROC_ROOT" \
  --split_dir "$SPLIT_DIR" \
  --train_split "$TRAIN_SPLIT" \
  --val_split "$VAL_SPLIT" \
  --train_fakes "$TRAIN_FAKES" \
  --val_fakes "$VAL_FAKES" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --save_path "$SAVE_PATH" \
  $EXTRA_ARGS
