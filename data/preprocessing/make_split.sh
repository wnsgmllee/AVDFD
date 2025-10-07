#!/usr/bin/env bash
set -euo pipefail

# =========== 사용자 설정 ===========
DATA_ROOT="."   # FakeAVCeleb 루트(지금 폴더가 루트면 ".")
OUT_DIR="/data/jhlee39/workspace/repos/OpenAVFF/data"

TRAIN_RATIO="0.75"     # 0.7 => 7:3
FAKE_MULT="3"         # fake = real x K (1:5 -> 5)

ONLY_RVFA="true"     # true면 RVFA(또는 Refine)만 사용, 자동 1:1
VC_MODE="true"       # true면 RVFA → RealVideo-FakeAudio-Refine

# CROSS_MANIP: "", "RVFA", "FVRA", "FVFA"
#   ""      : OFF
#   "RVFA"  : TRAIN은 RVFA 제외, TEST는 RVFA만 사용 (다른 값도 동일 규칙)
CROSS_MANIP="RVFA"

SEED="1337"
PYTHON_BIN="python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ==================================

ARGS=( "--data-root" "$DATA_ROOT"
       "--out-dir" "$OUT_DIR"
       "--train-ratio" "$TRAIN_RATIO"
       "--seed" "$SEED"
)

if [[ -n "$CROSS_MANIP" ]]; then
  ARGS+=( "--cross-manip" "$CROSS_MANIP" )
  echo "[INFO] CROSS_MANIP   = $CROSS_MANIP (TRAIN excludes it; TEST uses only it)"
else
  if [[ "$ONLY_RVFA" == "true" ]]; then
    ARGS+=( "--only-RVFA" )
    echo "[INFO] ONLY_RVFA     = true (ratio forced to 1:1)"
  else
    ARGS+=( "--fake-mult" "$FAKE_MULT" )
    echo "[INFO] FAKE_MULT     = $FAKE_MULT (fake = real x FAKE_MULT)"
  fi
fi

if [[ "$VC_MODE" == "true" ]]; then
  ARGS+=( "--vc" )
  echo "[INFO] VC_MODE       = true (use RealVideo-FakeAudio-Refine)"
else
  echo "[INFO] VC_MODE       = false"
fi

echo "[INFO] DATA_ROOT     = $DATA_ROOT"
echo "[INFO] OUT_DIR       = $OUT_DIR"
echo "[INFO] TRAIN_RATIO   = $TRAIN_RATIO"
echo "[INFO] ONLY_RVFA     = $ONLY_RVFA"
echo "[INFO] SEED          = $SEED"
echo

set -x
"$PYTHON_BIN" "$SCRIPT_DIR/make_split.py" "${ARGS[@]}"
set +x

echo
echo "[INFO] Listing OUT_DIR:"
ls -l "$OUT_DIR" || true

