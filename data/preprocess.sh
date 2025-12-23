#!/usr/bin/bash
#SBATCH -J AVDFD-preproc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 2-0
#SBATCH -o logs/preproc-%A.out
#SBATCH --nodelist=ariel-v8


DATA_ROOT="/data/jhlee39/workspace/repos/AVDFD/data/FakeAVCeleb_Refine/FakeAVCeleb_v1.2"
OUT_ROOT="/data/jhlee39/workspace/repos/AVDFD/data/FakeAVCeleb_Refine/preproc"

VERSIONS="ALL"          # or "RVRA,RVFA,FVRA,FVFA,RVFA-VC,RVFA-SVC"
CLIPS_PER_VIDEO=1
SEED=0
MODEL_ID="cvssp/audioldm2"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_DISABLE=1


python preprocess.py \
  --data_root "$DATA_ROOT" \
  --versions "$VERSIONS" \
  --out_root "$OUT_ROOT" \
  --clips_per_video "$CLIPS_PER_VIDEO" \
  --seed "$SEED" \
  --model_id "$MODEL_ID"
