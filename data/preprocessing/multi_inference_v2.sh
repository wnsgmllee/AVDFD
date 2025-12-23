#!/usr/bin/bash

#SBATCH -J seedVC
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 0-23
#SBATCH -o logs/slurm-%A.out
#SBATCH --nodelist=ariel-v6

# ==== 임시 작업 디렉토리(큰 디스크) ====
TMP_DIR="/data2/local_datasets/jhlee39/tmp"   # 존재하지 않으면 자동 생성됨
export TMPDIR="$TMP_DIR"

# ==== 사용자가 바꿀 곳 (필수 경로 3개) ====
SRC_ROOT="/data2/local_datasets/jhlee39/FakeAVCeleb_v1.2/RealVideo-RealAudio"
DST_ROOT="/data2/local_datasets/jhlee39/FakeAVCeleb_v1.2/RealVideo-FakeAudio-Refine"
VOX_ROOT="/data2/local_datasets/jhlee39/voxceleb_v2/data"

# ==== 모델 캐시를 ./models로 고정 (원하면 절대경로로) ====
LOCAL_MODELS="./models"
OFFLINE=false   # 완전 오프라인으로 돌리려면 true

# ==== 하이퍼파라미터 (원하면 조정) ====
DIFF_STEPS=25
LENGTH_ADJUST=1.0
COMPILE=false
INTEL_CFG=0.7
SIM_CFG=0.70
TOP_P=0.9
TEMP=1.0
REP=1.0
CONVERT_STYLE=false
ANON=false

# ==== 실행 ====
python multi_inference_v2.py \
  --src-root  "${SRC_ROOT}" \
  --dst-root  "${DST_ROOT}" \
  --vox-root  "${VOX_ROOT}" \
  --local-models-dir "${LOCAL_MODELS}" \
  --tmp-dir "${TMP_DIR}" \
  --offline "${OFFLINE}" \
  --diffusion-steps "${DIFF_STEPS}" \
  --length-adjust    "${LENGTH_ADJUST}" \
  --compile          "${COMPILE}" \
  --intelligibility-cfg-rate "${INTEL_CFG}" \
  --similarity-cfg-rate      "${SIM_CFG}" \
  --top-p "${TOP_P}" \
  --temperature "${TEMP}" \
  --repetition-penalty "${REP}" \
  --convert-style "${CONVERT_STYLE}" \
  --anonymization-only "${ANON}"
