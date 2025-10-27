#!/usr/bin/bash

# ==== 사용자 환경 설정 ====
DATA_ROOT="/data/jhlee39/workspace/repos/AVDFD/data/FakeAVCeleb_Refine/FakeAVCeleb_v1.2"
OUT_DIR="/data/jhlee39/workspace/repos/AVDFD/OpenAVFF/data"

# 새로 추가: out_dir 안의 하위 디렉토리명
DIR_NAME="CM_RVFA_harder_0.8"   # 비워두면 OUT_DIR에 바로 저장

TRAIN_RATIO="0.8"
FAKE_MULT="3"

ONLY_RVFA=""      # true면 RVFA만 사용(VC_MODE에 따라 경로 다름)
VC_MODE="harder"     # "false", "harder", or "hardest"

# cross_manip: RVFA / FVRA / FVFA (대소문자 무시). 빈 문자열이면 미사용.
# ONLY_RVFA와 동시 사용은 권장하지 않음(논리 충돌).
# 설정 시 train은 그 조작을 제외한 나머지들에서 샘플링, test는 그 조작에서만 샘플링
CROSS_MANIP="RVFA"

SEED="42"
PYTHON_BIN="python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=( "--data-root" "$DATA_ROOT"
       "--out-dir" "$OUT_DIR"
       "--train-ratio" "$TRAIN_RATIO"
       "--seed" "$SEED"
       "--vc" "$VC_MODE"
)

# 새로 추가: dir-name 전달
if [[ -n "$DIR_NAME" ]]; then
  ARGS+=( "--dir-name" "$DIR_NAME" )
fi

# cross_manip이 설정되면 그것을 우선, 아니면 ONLY_RVFA/FAKE_MULT 분기
if [[ -n "$CROSS_MANIP" ]]; then
  ARGS+=( "--cross-manip" "$CROSS_MANIP" )
else
  if [[ "$ONLY_RVFA" == "true" ]]; then
    ARGS+=( "--only-RVFA" )
  else
    ARGS+=( "--fake-mult" "$FAKE_MULT" )
  fi
fi

echo "[INFO] VC_MODE       = $VC_MODE"
echo "[INFO] CROSS_MANIP   = ${CROSS_MANIP:-None}"
echo "[INFO] ONLY_RVFA     = $ONLY_RVFA"
echo "[INFO] DATA_ROOT     = $DATA_ROOT"
echo "[INFO] OUT_DIR       = $OUT_DIR"
echo "[INFO] DIR_NAME      = ${DIR_NAME:-<none>}"
echo "[INFO] SEED          = $SEED"
echo

set -euo pipefail
set -x
"$PYTHON_BIN" "$SCRIPT_DIR/make_split.py" "${ARGS[@]}"
set +x

echo
echo "[INFO] Listing OUT_DIR:"
if [[ -n "$DIR_NAME" ]]; then
  ls -l "$OUT_DIR/$DIR_NAME" || true
else
  ls -l "$OUT_DIR" || true
fi
