#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, argparse, subprocess, random, tempfile, json, shutil
from pathlib import Path
from typing import Optional, List, Tuple

import yaml
import torch
import soundfile as sf
from tqdm import tqdm
from modules.commons import str2bool
from multiprocessing import Manager

DTYPE = torch.float16
vc_wrapper_v2 = None  # per-process singleton

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
def resolve_device(auto_rank: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n == 0:
            return torch.device("cpu")
        if auto_rank is None:
            return torch.device("cuda:0")
        auto_rank = max(0, min(n - 1, int(auto_rank)))
        return torch.device(f"cuda:{auto_rank}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# ─────────────────────────────────────────────
# HF cache
# ─────────────────────────────────────────────
def setup_local_hf_cache(local_models_dir: str, offline: bool = False):
    local = Path(local_models_dir).absolute()
    (local / "hub").mkdir(parents=True, exist_ok=True)
    (local / "transformers").mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(local)
    os.environ["HF_HUB_CACHE"] = str(local / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(local / "transformers")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.pop("HF_ENDPOINT", None)

# ─────────────────────────────────────────────
# Load V2 (seed-vc/inference_v2.py와 동일 흐름)
# ─────────────────────────────────────────────
def load_v2_models(args, device: torch.device):
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc = instantiate(cfg)
    vc.load_checkpoints(
        ar_checkpoint_path=args.ar_checkpoint_path,
        cfm_checkpoint_path=args.cfm_checkpoint_path
    )
    vc.to(device)
    vc.eval()
    vc.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=DTYPE, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        if hasattr(torch._inductor.config, "fx_graph_cache"):
            torch._inductor.config.fx_graph_cache = True
        vc.compile_ar()
    return vc

def convert_voice_v2(source_audio_path: str, target_audio_path: str, args, device: torch.device):
    """seed-vc inference_v2.py와 동일한 convert_voice_with_streaming 사용, 최종 버퍼만 취함."""
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args, device)

    gen = vc_wrapper_v2.convert_voice_with_streaming(
        source_audio_path=source_audio_path,
        target_audio_path=target_audio_path,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligebility_cfg_rate=args.intelligibility_cfg_rate,
        similarity_cfg_rate=args.similarity_cfg_rate,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        convert_style=args.convert_style,
        anonymization_only=args.anonymization_only,
        device=device,
        dtype=DTYPE,
        stream_output=True,
    )
    full = None
    for _, full in gen:
        pass
    return full  # (sr, np.ndarray) or None

# ─────────────────────────────────────────────
# ffmpeg Helpers (심플) + 0초 검증 유틸
# ─────────────────────────────────────────────
def ffprobe_audio_has_data(path: str) -> bool:
    """오디오 스트림 존재 & 길이>0 인지 빠르게 점검."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-select_streams", "a:0", path],
            capture_output=True, text=True, check=True
        )
        data = json.loads(proc.stdout.strip() or "{}")
        streams = data.get("streams", [])
        if not streams:
            return False
        s = streams[0]
        dur = float(s.get("duration", "0") or 0)
        if dur > 0:
            return True
        if int(s.get("nb_frames", "0") or 0) > 0:
            return True
        return False
    except Exception:
        return False

def ffmpeg_extract_wav(src_media: str, out_wav: str) -> None:
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "warning", "-y", "-i", src_media, "-vn", out_wav]
    subprocess.run(cmd, check=True)

def ffmpeg_mux_audio(video_mp4: str, new_wav: str, out_mp4: str) -> None:
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "warning", "-y",
        "-i", video_mp4, "-i", new_wav,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-q:a", "2",
        "-shortest", "-movflags", "+faststart",
        out_mp4
    ]
    subprocess.run(cmd, check=True)

# ─────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────
def id_from_mp4_path(p: str) -> str:
    # .../race/gender/id00076/file.mp4 → "id00076"
    return Path(p).parent.name

def list_all_wavs_under(vox_id_dir: str) -> List[str]:
    """id 루트 아래의 모든 하위 폴더를 재귀로 뒤져 *.wav 전체를 반환."""
    if not os.path.isdir(vox_id_dir):
        return []
    return [str(p) for p in Path(vox_id_dir).rglob("*.wav")]

def pick_valid_reference_wav(vox_id_dir: str) -> Optional[str]:
    """
    같은 id 내에서 '빈 파일이 아닌' 유효한 wav를 찾을 때까지 반복해서 시도.
    유효한 것이 하나도 없으면 None.
    """
    wavs = list_all_wavs_under(vox_id_dir)
    if not wavs:
        return None
    random.shuffle(wavs)
    for w in wavs:
        if ffprobe_audio_has_data(w):
            return w
    return None

def mirror_dst_path(src_root: str, dst_root: str, abs_src_mp4: str) -> str:
    rel = os.path.relpath(abs_src_mp4, start=src_root)
    return os.path.join(dst_root, rel)

# ─────────────────────────────────────────────
# Core per-file (OK / IVC만, SKIP 제거)
# ─────────────────────────────────────────────
def try_vc(source_wav: str, target_wav: str, args, device: torch.device) -> Optional[Tuple[int, "np.ndarray"]]:
    """VC 한 번 시도 (실패 시 None)"""
    try:
        out = convert_voice_v2(source_wav, target_wav, args, device)
        return out
    except Exception:
        return None

def do_one(mp4_path: str, args, device: torch.device, vox_root: str, src_root: str, dst_root: str,
           identity_list) -> Tuple[str, str, str, str]:
    """
    return: (status, rel_path, id, info)  // status in {"OK","IVC"}
    identity_list: identity 경로(VC 실패/0초/참조불가 등으로 IVC 처리된 샘플)의 src 상대경로 저장
    """
    rel = os.path.relpath(mp4_path, start=src_root)
    this_id = id_from_mp4_path(mp4_path)
    vox_id_dir = os.path.join(vox_root, this_id)

    ref_wav = pick_valid_reference_wav(vox_id_dir)  # 유효 ref 탐색 (없으면 None)

    tmp_root = args.tmp_dir if getattr(args, "tmp_dir", None) else None
    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="vc2_", dir=tmp_root)

    try:
        # 1) 원본에서 오디오 추출
        target_wav = os.path.join(tmp, "target.wav")
        ffmpeg_extract_wav(mp4_path, target_wav)
        src_has = ffprobe_audio_has_data(target_wav)

        # 2) 시나리오 분기
        used_identity = False
        converted = None

        # (A) 정상 경로: ref가 있고 VC 성공하면 OK
        if src_has and ref_wav is not None:
            converted = try_vc(target_wav, ref_wav, args, device)

        # (B) ref가 없거나 VC가 실패/0초 → identity 시도
        need_identity = (not src_has) or (ref_wav is None) or (converted is None) \
                        or (converted and converted[1] is not None and len(converted[1]) == 0)

        if need_identity:
            if src_has:
                # 1) identity VC 먼저 시도
                ident = try_vc(target_wav, target_wav, args, device)
                if ident is not None and ident[1] is not None and len(ident[1]) > 0:
                    converted = ident
                    used_identity = True
                else:
                    # 2) VC 자체를 생략하고 원본 오디오를 그대로 사용 (항상 길이>0 보장됨)
                    #    이 경우도 IVC로 분류
                    sr = None
                    try:
                        # soundfile로 sr 알아내기 (없어도 ffmpeg가 처리하긴 함)
                        info = sf.info(target_wav)
                        sr = info.samplerate
                    except Exception:
                        pass
                    # converted 형식 맞춰 저장용으로 구성
                    if sr is None:
                        sr = 44100  # 정보 못 얻으면 임의값, 실제 mux는 ffmpeg가 처리
                    data, _sr = sf.read(target_wav, dtype="float32", always_2d=False)
                    converted = (sr, data)
                    used_identity = True
            else:
                # source 자체에 유효 오디오가 없는 드문 케이스:
                # 결과를 만들 수 없으므로, 영상만 복사(무오디오)하여 IVC로 표기
                out_mp4 = mirror_dst_path(src_root, dst_root, mp4_path)
                Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    "ffmpeg", "-v", "warning", "-y",
                    "-i", mp4_path, "-c:v", "copy", "-an",
                    "-movflags", "+faststart", out_mp4
                ], check=True)
                identity_list.append(rel + "  [no-src-audio]")
                return ("IVC", rel, this_id, os.path.relpath(out_mp4, dst_root))

        # 여기까지 오면 converted는 반드시 존재
        save_sr, audio_arr = converted

        # 3) 결과 저장
        gen_wav = os.path.join(tmp, "converted.wav")
        sf.write(gen_wav, audio_arr, save_sr)

        # 4) 비디오 + 오디오 합치기
        out_mp4 = mirror_dst_path(src_root, dst_root, mp4_path)
        ffmpeg_mux_audio(mp4_path, gen_wav, out_mp4)

        if used_identity:
            identity_list.append(rel)
            return ("IVC", rel, this_id, os.path.relpath(out_mp4, dst_root))
        else:
            return ("OK", rel, this_id, os.path.relpath(out_mp4, dst_root))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ─────────────────────────────────────────────
# Multi-GPU
# ─────────────────────────────────────────────
def shard(lst: List[str], n: int, k: int) -> List[str]:
    if n <= 1:
        return lst
    return [p for i, p in enumerate(lst) if (i % n) == k]

def worker(rank: int, files: List[str], args, src_root: str, dst_root: str, vox_root: str,
           identity_list, counters):
    """
    counters: dict-like Manager 객체 {"OK": int, "IVC": int}
    """
    device = resolve_device(rank if torch.cuda.is_available() else None)
    print(f"[GPU{rank if torch.cuda.is_available() else 'CPU'}] start {len(files)} files on {device}", flush=True)

    ok = ivc = 0
    pbar = tqdm(files, desc=f"GPU{rank} ({device})", unit="file", dynamic_ncols=True, mininterval=0.2)
    for f in pbar:
        status, rel, this_id, info = do_one(f, args, device, vox_root, src_root, dst_root, identity_list)
        if status == "OK":
            ok += 1
        else:  # "IVC"
            ivc += 1
        pbar.set_postfix_str(f"OK {ok} | IVC {ivc}", refresh=False)
    pbar.close()

    counters["OK"] += ok
    counters["IVC"] += ivc
    print(f"[DONE][{rank}] OK={ok} IVC={ivc}", flush=True)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    import multiprocessing as mp

    p = argparse.ArgumentParser(description="Multi-Inference V2 over dataset (ref retry + identity only; OK/IVC)")
    # 필수 경로
    p.add_argument("--src-root", required=True, type=str, help="입력 mp4 루트")
    p.add_argument("--dst-root", required=True, type=str, help="출력 mp4 루트(원본과 동일 트리)")
    p.add_argument("--vox-root", required=True, type=str, help="VoxCeleb2 루트(동일 id에서 ref 선택)")

    # seed-vc v2 인자 (원형 유지)
    p.add_argument("--diffusion-steps", type=int, default=30)
    p.add_argument("--length-adjust", type=float, default=1.0)
    p.add_argument("--compile", type=str2bool, default=False)
    p.add_argument("--intelligibility-cfg-rate", type=float, default=0.7)
    p.add_argument("--similarity-cfg-rate", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--convert-style", type=str2bool, default=False)
    p.add_argument("--anonymization-only", type=str2bool, default=False)
    p.add_argument("--ar-checkpoint-path", type=str, default=None)
    p.add_argument("--cfm-checkpoint-path", type=str, default=None)

    # 모델 캐시/오프라인
    p.add_argument("--local-models-dir", type=str, default="./models")
    p.add_argument("--offline", type=str2bool, default=False)

    # 임시 디렉토리
    p.add_argument("--tmp-dir", type=str, default=None)

    args = p.parse_args()

    setup_local_hf_cache(args.local_models_dir, offline=args.offline)

    # CUDA fork 이슈 회피
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    src_root = os.path.abspath(args.src_root)
    dst_root = os.path.abspath(args.dst_root)
    vox_root = os.path.abspath(args.vox_root)

    files = [str(p) for p in Path(src_root).rglob("*.mp4")]
    files.sort()
    if not files:
        print(f"[ERR] no mp4 found under {src_root}", file=sys.stderr)
        sys.exit(1)

    with Manager() as manager:
        identity_list = manager.list()
        counters = manager.dict()
        counters["OK"] = 0
        counters["IVC"] = 0

        try:
            ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            ngpu = 0

        if ngpu <= 1:
            worker(rank=0, files=files, args=args, src_root=src_root, dst_root=dst_root,
                   vox_root=vox_root, identity_list=identity_list, counters=counters)
        else:
            procs = []
            for rank in range(ngpu):
                shard_list = files[rank::ngpu]
                if not shard_list:
                    continue
                p_ = mp.Process(target=worker,
                                args=(rank, shard_list, args, src_root, dst_root, vox_root,
                                      identity_list, counters))
                p_.start()
                procs.append(p_)
            for p_ in procs:
                p_.join()

        # 최종 요약 출력
        print("\n================ SUMMARY ================")
        print(f"OK : {int(counters['OK'])}")
        print(f"IVC: {int(counters['IVC'])}")

        if len(identity_list) > 0:
            print("\n[IDENTITY VC APPLIED (no valid ref / VC failed / used original audio)]")
            for rel in list(identity_list):
                print(rel)
        print("=========================================\n")

if __name__ == "__main__":
    main()
