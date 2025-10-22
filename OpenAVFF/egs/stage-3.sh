#!/usr/bin/bash
#SBATCH -J AVFF
#SBATCH --gres=gpu:4                 # 원하는 GPU 수 (잡 제출 시 조정 가능)
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 5-23
#SBATCH -o logs/slurm-%A.out
#SBATCH --nodelist=ariel-v6

set -euo pipefail

# =========================
# 공통 환경 안정화
# =========================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_CACHE_DISABLE=1

# 로그 디렉토리
mkdir -p logs

# =========================
# 하이퍼파라미터 (보수적 기본값)
# =========================
contrast_loss_weight=0.01
mae_loss_weight=1.0
norm_pix_loss=True

# Stage-2 -> Stage-3 변환 결과(백본만 이식, 헤드는 랜덤)
pretrain_path=../checkpoints/stage3_init_from_stage2.pth

# LR: 백본/헤드 분리
lr=3e-5
head_lr=50

epoch=10000
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa_start=1
wa_end=10

dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True

# CSV 경로
tr_data=../data/trainset.csv
te_data=../data/testset.csv

# 저장 경로 (모델은 저장하지 않고, 로그만 저장)
save_dir=../checkpoints/stage3_custom
mkdir -p "${save_dir}"

# =========================
# GPU/워커 자동 설정
# =========================
NGPUS=${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}}
NGPUS=$(echo "$NGPUS" | sed 's/[^0-9].*$//')
if [[ -z "${NGPUS}" || "${NGPUS}" -lt 1 ]]; then
  NGPUS=1
fi

# GPU당 배치/워커
BATCH_PER_GPU=2
NUM_WORKERS_PER_GPU=6

# =========================
# 실행 커맨드 공통 옵션
# =========================
COMMON_ARGS=(
  --data-train "${tr_data}"
  --data-val   "${te_data}"
  --save-dir   "${save_dir}"
  --n_classes  2
  --lr         "${lr}"
  --head_lr    "${head_lr}"
  --n-epochs   "${epoch}"
  --batch-size "${BATCH_PER_GPU}"
  --num_workers "${NUM_WORKERS_PER_GPU}"
  --lrscheduler_start "${lrscheduler_start}"
  --lrscheduler_decay "${lrscheduler_decay}"
  --lrscheduler_step  "${lrscheduler_step}"
  --dataset_mean "${dataset_mean}"
  --dataset_std  "${dataset_std}"
  --target_length "${target_length}"
  --noise "${noise}"
  --lr_adapt False
  --norm_pix_loss "${norm_pix_loss}"
  --mae_loss_weight "${mae_loss_weight}"
  --contrast_loss_weight "${contrast_loss_weight}"
  --loss BCE
  --metrics mAP
  --warmup True
  --wa_start "${wa_start}"
  --wa_end   "${wa_end}"
  --pretrain_path "${pretrain_path}"
)

echo "Detected GPUs: ${NGPUS}"
echo "Batch per GPU: ${BATCH_PER_GPU}  |  Num workers per GPU: ${NUM_WORKERS_PER_GPU}"
echo "Logs to: ${save_dir}"
echo "Starting training (no checkpoint saving)..."

# =========================
# 실행 (싱글 vs 멀티 자동 분기)
# =========================
if [[ "${NGPUS}" -ge 2 ]]; then
  torchrun --standalone --nproc_per_node="${NGPUS}" ../src/run_ft.py \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${save_dir}/train.log"
else
  python -W ignore ../src/run_ft.py \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${save_dir}/train.log"
fi
