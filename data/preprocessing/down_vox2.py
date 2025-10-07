"""
vox2_wav_from_hf_simple.py

- 🤗 Datasets(streaming)으로 gaunernst/voxceleb2-dev-wds를 로드
- 각 샘플의 m4a 오디오만 추출하여 __key__ 기반 원래 폴더 구조로 *.wav 저장
- 병렬 변환: ffmpeg 프로세스를 CPU 코어 수에 맞춰 자동 병렬 실행
- 진행 상황: tqdm
- 기본값:
    * 샘플레이트, 채널수: 원본 유지
    * 덮어쓰기 안 함(이미 존재하면 스킵)
    * in-flight 작업 수: CPU 코어 수 * 4 (메모리 과점유 방지용 완충)
- 필요 인자:
    * --out-dir : 출력 루트 디렉터리 (필수)
"""


import argparse
from pathlib import Path
from typing import Optional, Deque
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from tqdm import tqdm
from datasets import load_dataset

def cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1

def to_bytes(x) -> bytes:
    """datasets의 필드가 bytes / file-like / 경로 등일 수 있어 범용 변환."""
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    try:
        return x.read()  # StreamingBytes 등 file-like
    except Exception:
        p = Path(str(x))
        with p.open("rb") as f:
            return f.read()

def ffmpeg_m4a_to_wav(m4a_bytes: bytes, wav_path: Path,
                      sr: Optional[int] = None, mono: bool = False) -> None:
    """ffmpeg로 m4a(bytes) → wav(file). 기본은 원본 SR/채널 유지."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0"]
    if mono:
        cmd += ["-ac", "1"]
    if sr is not None:
        cmd += ["-ar", str(sr)]
    cmd += [str(wav_path)]
    proc = subprocess.run(cmd, input=m4a_bytes,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

def build_allowed_ids(id_root: Path) -> set[str]:
    """
    RealVideo-RealAudio 트리에서 허용할 id 집합 생성.
    폴더 구조: <race>/<gender>/<idXXXXXX>/*.mp4
    """
    allowed = set()
    # race/*/gender/*/id*/  깊이 3의 디렉토리명이 idXXXXXX
    for id_dir in id_root.glob("*/*/id*"):
        if id_dir.is_dir():
            allowed.add(id_dir.name)
    return allowed

def id_from_key(key: str) -> str:
    """
    datasets 샘플 key 예: 'id02139/yCPbcLeT5SI/00147'
    → 맨 앞 컴포넌트가 id
    """
    return key.split("/", 1)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path, help="WAV 저장 루트 디렉터리")
    ap.add_argument("--id-root", required=True, type=Path,
                    help="허용할 id를 추출할 RealVideo-RealAudio 루트 (race/gender/idXXXX 구조)")
    # 아래 인자들은 기본값/자동감지로 돌리되, 혹시 바꾸고 싶을 때만 사용
    ap.add_argument("--repo", default="gaunernst/voxceleb2-dev-wds")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-workers", type=int, default=cpu_count(),
                    help="동시 변환 작업 수(기본: CPU 코어 수)")
    ap.add_argument("--max-in-flight", type=int, default=max(4, (cpu_count() or 1) * 4),
                    help="제출해둘 최대 작업 수(기본: CPU*4)")
    ap.add_argument("--overwrite", action="store_true", help="기존 wav도 다시 만들기")
    # 보통은 그대로 두세요(원본 유지). 필요하면 --sr 16000 --mono 등 추가로 사용 가능.
    ap.add_argument("--sr", type=int, default=None, help="리샘플링 샘플레이트")
    ap.add_argument("--mono", action="store_true", help="모노 변환")
    ap.add_argument("--expected-total", type=int, default=None,
                    help="tqdm total (모르면 생략)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 허용 id 집합 생성
    allowed_ids = build_allowed_ids(args.id_root)
    if not allowed_ids:
        raise RuntimeError(f"[ERR] 허용 id를 한 개도 찾지 못했습니다: {args.id_root}")

    # 🤗 Datasets: 스트리밍 로드(전체 사전 다운로드 없음)
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    processed = skipped = errors = ignored = 0
    pool = ThreadPoolExecutor(max_workers=args.num_workers)
    in_flight: Deque = deque()

    def submit(key, m4a_obj):
        """in-flight 상한을 유지하며 작업 제출."""
        while len(in_flight) >= args.max_in_flight:
            fut = in_flight.popleft()
            yield fut
        in_flight.append(pool.submit(process_one, key, m4a_obj))

    def process_one(key, m4a_obj):
        """한 샘플 처리(존재 시 스킵; ffmpeg 변환)."""
        wav_path = (args.out_dir / key).with_suffix(".wav")
        if wav_path.exists() and not args.overwrite:
            return "skip"
        m4a_bytes = to_bytes(m4a_obj)
        ffmpeg_m4a_to_wav(m4a_bytes, wav_path, sr=args.sr, mono=args.mono)
        return "ok"

    try:
        with tqdm(total=args.expected_total, unit="sample", smoothing=0.02) as bar:
            for sample in ds:
                key = sample["__key__"]   # 예: id02139/yCPbcLeT5SI/00147
                m4a_obj = sample["m4a"]

                # ✅ 허용 id 필터링: 허용되지 않으면 즉시 무시(저장 X)
                this_id = id_from_key(key)
                if this_id not in allowed_ids:
                    ignored += 1
                    bar.update(1)
                    continue

                for done in submit(key, m4a_obj):  # backpressure
                    try:
                        r = done.result()
                        if r == "ok":
                            processed += 1
                        elif r == "skip":
                            skipped += 1
                        else:
                            errors += 1
                    except Exception:
                        errors += 1
                    bar.update(1)

            # 남은 작업 정리
            while in_flight:
                fut = in_flight.popleft()
                try:
                    r = fut.result()
                    if r == "ok":
                        processed += 1
                    elif r == "skip":
                        skipped += 1
                    else:
                        errors += 1
                except Exception:
                    errors += 1
                bar.update(1)
    finally:
        pool.shutdown(wait=True)

    print(f"\n완료: 처리 {processed}개, 스킵 {skipped}개, 무시(비허용 id) {ignored}개, 오류 {errors}개")

if __name__ == "__main__":
    main()

