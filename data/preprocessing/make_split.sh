#!/usr/bin/bash

DATA_ROOT="/data/jhlee39/workspace/repos/AVDFD/data/FakeAVCeleb_Refine/FakeAVCeleb_v1.2"
OUT_DIR="/data/jhlee39/workspace/repos/OpenAVFF/data"
TRAIN_RATIO="0.75"
FAKE_MULT="3"
ONLY_RVFA="true"
VC_MODE="hardest"  # "false", "harder", or "hardest"
CROSS_MANIP=""
SEED="1337"
PYTHON_BIN="python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=( "--data-root" "$DATA_ROOT"
       "--out-dir" "$OUT_DIR"
       "--train-ratio" "$TRAIN_RATIO"
       "--seed" "$SEED"
       "--vc" "$VC_MODE"
)

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
echo "[INFO] CROSS_MANIP   = $CROSS_MANIP"
echo "[INFO] DATA_ROOT     = $DATA_ROOT"
echo "[INFO] OUT_DIR       = $OUT_DIR"
echo "[INFO] SEED          = $SEED"
echo

set -x
"$PYTHON_BIN" "$SCRIPT_DIR/make_split.py" "${ARGS[@]}"
set +x

echo "[INFO] Listing OUT_DIR:"
ls -l "$OUT_DIR" || true
