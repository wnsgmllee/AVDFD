# Harder/Hardest sample 생성용

import os, sys, argparse, subprocess, random, tempfile, json, shutil, hashlib
from pathlib import Path
from typing import Optional, List, Tuple

import yaml
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from multiprocessing import Manager

# 외부 패키지
# speechbrain (ECAPA) 임베딩용
from speechbrain.pretrained import EncoderClassifier

# 프로젝트 유틸
from modules.commons import str2bool

DTYPE = torch.float16

# per-process singletons
vc_wrapper_v2 = None
ecapa_encoder: Optional[EncoderClassifier] = None

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
# HF cache (for seed-vc; speechbrain는 자체 다운로드 경로 사용)
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
# Seed-VC v2 로딩 (원본 흐름 유지)
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

def get_vc_wrapper(args, device: torch.device):
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args, device)
    return vc_wrapper_v2

def convert_voice_v2(source_audio_path: str, target_audio_path: str, args, device: torch.device):
    """seed-vc inference_v2.py의 스트리밍 호출. 최종 버퍼 반환."""
    vc = get_vc_wrapper(args, device)
    gen = vc.convert_voice_with_streaming(
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
# ffmpeg Helpers + 0초 검증/정규화 유틸
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

def ffmpeg_resample_16k_mono(in_wav: str, out_wav: str) -> None:
    """임베딩/동일성 비교를 위한 표준화(16kHz mono PCM)."""
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", in_wav, "-ac", "1", "-ar", "16000", out_wav]
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

def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

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

def mirror_dst_path(src_root: str, dst_root: str, abs_src_mp4: str, tier: str) -> str:
    """tier: 'hardest' or 'harder'"""
    rel = os.path.relpath(abs_src_mp4, start=src_root)
    return os.path.join(dst_root, tier, rel)

# ─────────────────────────────────────────────
# ECAPA 임베딩/유사도
# ─────────────────────────────────────────────
def get_ecapa(device: torch.device) -> EncoderClassifier:
    global ecapa_encoder
    if ecapa_encoder is None:
        # speechbrain은 자체 캐시 디렉토리를 사용.
        ecapa_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)}
        )
    return ecapa_encoder

def load_mono16k_tensor(wav16k_path: str, device: torch.device) -> torch.Tensor:
    """soundfile로 로딩 (16k mono 가정). 출력 shape: (1, T) float32."""
    data, sr = sf.read(wav16k_path, dtype="float32", always_2d=False)
    if sr != 16000:
        raise RuntimeError(f"Expected 16k, got {sr}")
    if data.ndim == 1:
        x = data[None, ...]  # (1, T)
    else:
        x = data.mean(axis=1, keepdims=True).T  # (1, T)
    return torch.from_numpy(x).to(device)

def ecapa_embedding(wav16k_path: str, device: torch.device) -> torch.Tensor:
    enc = get_ecapa(device)
    x = load_mono16k_tensor(wav16k_path, device)
    with torch.no_grad():
        emb = enc.encode_batch(x)  # (1, D)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0)  # (D,)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.clamp(torch.sum(a * b), -1.0, 1.0).item())

# ─────────────────────────────────────────────
# Core VC helpers
# ─────────────────────────────────────────────
def try_vc(source_wav: str, target_wav: str, args, device: torch.device) -> Optional[Tuple[int, "np.ndarray"]]:
    """VC 한 번 시도 (실패 시 None)"""
    try:
        out = convert_voice_v2(source_wav, target_wav, args, device)
        return out
    except Exception:
        return None

def ensure_audio_tuple_from_wav(path: str) -> Tuple[int, np.ndarray]:
    """wav 파일을 (sr, np.ndarray)로 로딩. 실패 시 예외."""
    info = sf.info(path)
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data is None or (hasattr(data, "__len__") and len(data) == 0):
        raise RuntimeError("empty audio")
    return (sr, data)

# ─────────────────────────────────────────────
# HARDER: 같은 id 내에서 ref 후보 검색 (동일음성 제외) → 최고 유사도 1개 선택
# ─────────────────────────────────────────────
def pick_best_ref_by_ecapa(vox_id_dir: str, target_wav: str, tmp_dir: str, device: torch.device) -> Optional[str]:
    """
    - target_wav: 원본 MP4에서 추출한 wav (아무 SR/채널 상관 없음)
    - 동일성 제외: target_wav와 동일한 16k mono 파형(sha1)이면 제외
    - 유사도: ECAPA 임베딩 코사인 유사도 최대 1개 선택
    """
    candidates = list_all_wavs_under(vox_id_dir)
    if not candidates:
        return None

    # 표준화된 16k mono 파형 및 해시
    target_std = os.path.join(tmp_dir, "target_16k.wav")
    ffmpeg_resample_16k_mono(target_wav, target_std)
    target_hash = sha1_of_file(target_std)
    target_emb = ecapa_embedding(target_std, device)

    best_path = None
    best_sim = -2.0  # cosine lower bound

    # 후보 루프
    for w in candidates:
        try:
            if not ffprobe_audio_has_data(w):
                continue
            cand_std = os.path.join(tmp_dir, "__cand_16k.wav")
            ffmpeg_resample_16k_mono(w, cand_std)
            cand_hash = sha1_of_file(cand_std)
            # 동일 음성(완전 동일 파형) 제외
            if cand_hash == target_hash:
                continue
            cand_emb = ecapa_embedding(cand_std, device)
            sim = cosine_sim(target_emb, cand_emb)
            if sim > best_sim:
                best_sim = sim
                best_path = w
        except Exception:
            continue

    return best_path

# ─────────────────────────────────────────────
# 한 파일 처리: hardest + harder 모두 생성
# ─────────────────────────────────────────────
def do_one_file(mp4_path: str, args, device: torch.device, vox_root: str, src_root: str, dst_root: str,
                identity_list_hardest, identity_list_harder):
    """
    반환값 없음. 각 케이스별로 mux 및 카운터 반영은 worker에서 처리.
    identity_list_*: identity VC 또는 fallback(원본/무오디오) 사용된 상대경로 목록
    """
    rel = os.path.relpath(mp4_path, start=src_root)
    this_id = id_from_mp4_path(mp4_path)
    vox_id_dir = os.path.join(vox_root, this_id)

    tmp_root = args.tmp_dir if getattr(args, "tmp_dir", None) else None
    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="vc2_", dir=tmp_root)

    try:
        # 1) 원본에서 오디오 추출
        target_wav = os.path.join(tmp, "target.wav")
        ffmpeg_extract_wav(mp4_path, target_wav)
        src_has = ffprobe_audio_has_data(target_wav)

        # =======================
        # A) HARDEST (identity)
        # =======================
        hardest_used_identity = True  # identity 기준이므로 True (성공/실패는 아래에서 보정)
        hardest_converted = None
        out_mp4_hardest = mirror_dst_path(src_root, dst_root, mp4_path, tier="hardest")

        if src_has:
            # Identity VC
            ident = try_vc(target_wav, target_wav, args, device)
            if ident is not None and ident[1] is not None and len(ident[1]) > 0:
                hardest_converted = ident
            else:
                # Identity VC 실패 → 원본 오디오를 그대로 사용
                try:
                    sr, data = ensure_audio_tuple_from_wav(target_wav)
                    hardest_converted = (sr, data)
                except Exception:
                    # 아주 드물게 원본 로딩 실패 시 무오디오로 복사
                    Path(out_mp4_hardest).parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run([
                        "ffmpeg", "-v", "warning", "-y",
                        "-i", mp4_path, "-c:v", "copy", "-an",
                        "-movflags", "+faststart", out_mp4_hardest
                    ], check=True)
                    identity_list_hardest.append(rel + "  [identity->no-audio]")
                    out_mp4_hardest = None  # 이미 저장 완료
        else:
            # 소스에 오디오가 없음 → 무오디오 복사
            Path(out_mp4_hardest).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "ffmpeg", "-v", "warning", "-y",
                "-i", mp4_path, "-c:v", "copy", "-an",
                "-movflags", "+faststart", out_mp4_hardest
            ], check=True)
            identity_list_hardest.append(rel + "  [no-src-audio]")
            out_mp4_hardest = None  # 이미 저장 완료

        if out_mp4_hardest is not None and hardest_converted is not None:
            gen_wav = os.path.join(tmp, "hardest.wav")
            sf.write(gen_wav, hardest_converted[1], hardest_converted[0])
            ffmpeg_mux_audio(mp4_path, gen_wav, out_mp4_hardest)

        # =======================
        # B) HARDER (best-ref by ECAPA)
        # =======================
        harder_used_identity = False
        harder_converted = None
        out_mp4_harder = mirror_dst_path(src_root, dst_root, mp4_path, tier="harder")

        if src_has:
            best_ref = pick_best_ref_by_ecapa(vox_id_dir, target_wav, tmp, device)
            if best_ref is not None:
                vc_out = try_vc(target_wav, best_ref, args, device)
                if vc_out is not None and vc_out[1] is not None and len(vc_out[1]) > 0:
                    harder_converted = vc_out
                else:
                    # ref VC 실패 → identity로 보정
                    ident = try_vc(target_wav, target_wav, args, device)
                    if ident is not None and ident[1] is not None and len(ident[1]) > 0:
                        harder_converted = ident
                        harder_used_identity = True
                    else:
                        # 최후: 원본 오디오 사용
                        sr, data = ensure_audio_tuple_from_wav(target_wav)
                        harder_converted = (sr, data)
                        harder_used_identity = True
            else:
                # ref 없음 → identity/원본으로 대응
                ident = try_vc(target_wav, target_wav, args, device)
                if ident is not None and ident[1] is not None and len(ident[1]) > 0:
                    harder_converted = ident
                    harder_used_identity = True
                else:
                    sr, data = ensure_audio_tuple_from_wav(target_wav)
                    harder_converted = (sr, data)
                    harder_used_identity = True
        else:
            # 소스에 오디오가 없음 → 무오디오 복사
            Path(out_mp4_harder).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "ffmpeg", "-v", "warning", "-y",
                "-i", mp4_path, "-c:v", "copy", "-an",
                "-movflags", "+faststart", out_mp4_harder
            ], check=True)
            identity_list_harder.append(rel + "  [no-src-audio]")
            out_mp4_harder = None

        if out_mp4_harder is not None and harder_converted is not None:
            gen_wav = os.path.join(tmp, "harder.wav")
            sf.write(gen_wav, harder_converted[1], harder_converted[0])
            ffmpeg_mux_audio(mp4_path, gen_wav, out_mp4_harder)
            if harder_used_identity:
                identity_list_harder.append(rel)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ─────────────────────────────────────────────
# Multi-GPU
# ─────────────────────────────────────────────
def worker(rank: int, files: List[str], args, src_root: str, dst_root: str, vox_root: str,
           identity_list_hardest, identity_list_harder, counters):
    """
    counters: dict-like Manager 객체 {"hardest": int, "harder": int}
    """
    device = resolve_device(rank if torch.cuda.is_available() else None)
    print(f"[GPU{rank if torch.cuda.is_available() else 'CPU'}] start {len(files)} files on {device}", flush=True)

    done_hardest = done_harder = 0
    pbar = tqdm(files, desc=f"GPU{rank} ({device})", unit="file", dynamic_ncols=True, mininterval=0.2)
    for f in pbar:
        do_one_file(f, args, device, vox_root, src_root, dst_root, identity_list_hardest, identity_list_harder)
        # 두 결과 모두 생성(무오디오 복사 포함)이므로 카운트를 +1씩
        done_hardest += 1
        done_harder += 1
        pbar.set_postfix_str(f"hardest {done_hardest} | harder {done_harder}", refresh=False)
    pbar.close()

    counters["hardest"] += done_hardest
    counters["harder"] += done_harder
    print(f"[DONE][{rank}] hardest={done_hardest} harder={done_harder}", flush=True)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    import multiprocessing as mp

    p = argparse.ArgumentParser(description="Multi-Inference V2 (hardest: identity, harder: best-ref by ECAPA; both outputs)")
    # 필수 경로
    p.add_argument("--src-root", required=True, type=str, help="입력 mp4 루트")
    p.add_argument("--dst-root", required=True, type=str, help="출력 mp4 루트(하위에 harder/hardest 생성)")
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

    # 모델 캐시/오프라인(seed-vc용)
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

    # 하위 폴더 미리 생성(harder/hardest)
    Path(os.path.join(dst_root, "hardest")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dst_root, "harder")).mkdir(parents=True, exist_ok=True)

    with Manager() as manager:
        identity_list_hardest = manager.list()
        identity_list_harder = manager.list()
        counters = manager.dict()
        counters["hardest"] = 0
        counters["harder"] = 0

        try:
            ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            ngpu = 0

        if ngpu <= 1:
            worker(rank=0, files=files, args=args, src_root=src_root, dst_root=dst_root,
                   vox_root=vox_root, identity_list_hardest=identity_list_hardest,
                   identity_list_harder=identity_list_harder, counters=counters)
        else:
            procs = []
            for rank in range(ngpu):
                shard_list = files[rank::ngpu]
                if not shard_list:
                    continue
                p_ = mp.Process(target=worker,
                                args=(rank, shard_list, args, src_root, dst_root, vox_root,
                                      identity_list_hardest, identity_list_harder, counters))
                p_.start()
                procs.append(p_)
            for p_ in procs:
                p_.join()

        # 최종 요약 출력
        print("\n================ SUMMARY ================")
        print(f"hardest : {int(counters['hardest'])}")
        print(f"harder  : {int(counters['harder'])}")

        if len(identity_list_hardest) > 0:
            print("\n[HARDEST: identity/fallback used]")
            for rel in list(identity_list_hardest):
                print(rel)

        if len(identity_list_harder) > 0:
            print("\n[HARDER: identity/fallback used (no ref / VC failed / identical filtered)]")
            for rel in list(identity_list_harder):
                print(rel)
        print("=========================================\n")

if __name__ == "__main__":
    main()
