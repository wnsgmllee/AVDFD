# preprocess.py
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from decord import VideoReader, cpu
import torchvision.transforms as T
import PIL.Image as Image
from tqdm import tqdm

# ---------------------------
# FakeAVCeleb version mapping
# ---------------------------
VERSION_DIR = {
    "RVRA": "RealVideo-RealAudio",
    "RVFA": "RealVideo-FakeAudio",
    "FVRA": "FakeVideo-RealAudio",
    "FVFA": "FakeVideo-FakeAudio",
    "RVFA-VC": "RealVideo-FakeAudio-VoiceClone",
    "RVFA-SVC": "RealVideo-FakeAudio-SelfVoiceClone",
}
ALL_VERSIONS = list(VERSION_DIR.keys())
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")

# ---------------------------
# LRS3-style preprocessing constants
# ---------------------------
SAMPLE_RATE = 16000
CLIP_SEC = 3.2

N_MELS_LRS3 = 128
WIN_LENGTH = int(0.016 * SAMPLE_RATE)   # 16 ms
HOP_LENGTH = int(0.004 * SAMPLE_RATE)   # 4 ms
N_FFT = 512                             # filterbank warning 완화
MEL_FRAMES_LRS3 = 768

N_FRAMES = 16
FRAME_SIZE = 224

# ---------------------------
# Mel spectrogram (LRS3 settings)
# ---------------------------
def make_lrs3_mel_transform():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window_fn=torch.hamming_window,
        n_mels=N_MELS_LRS3,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        mel_scale="htk",
    )

def wav_from_video(video_path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(video_path))
    if wav.ndim == 2:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze(0)

def lrs3_mel_clip(wav: torch.Tensor, start_sec: float, mel_transform) -> torch.Tensor:
    s = int(start_sec * SAMPLE_RATE)
    e = s + int(CLIP_SEC * SAMPLE_RATE)
    if e > wav.numel():
        wav_seg = F.pad(wav[s:], (0, e - wav.numel()))
    else:
        wav_seg = wav[s:e]

    mel = mel_transform(wav_seg.unsqueeze(0)).squeeze(0)  # [128, t]
    mel = torch.log(mel + 1e-6)

    t = mel.shape[1]
    if t < MEL_FRAMES_LRS3:
        mel = F.pad(mel, (0, MEL_FRAMES_LRS3 - t))
    elif t > MEL_FRAMES_LRS3:
        mel = mel[:, :MEL_FRAMES_LRS3]

    # 논문 조건: normalize (clip-wise z-score)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel  # [128,768]

# ---------------------------
# Frame sampling (LRS3)
# ---------------------------
def sample_clip_indices(total_len: int, fps: float):
    clip_len_frames = int(CLIP_SEC * fps)
    if total_len <= clip_len_frames:
        return 0, total_len
    start = random.randint(0, total_len - clip_len_frames)
    end = start + clip_len_frames
    return start, end

def sample_16_quartile(start: int, end: int):
    L = end - start
    slice_len = max(1, L // 8)
    idxs = []
    for s in range(8):
        s0 = start + s * slice_len
        s1 = min(end, s0 + slice_len)
        seg_len = max(1, s1 - s0)
        q1 = s0 + int(0.25 * seg_len)
        q3 = s0 + int(0.75 * seg_len)
        idxs.extend([q1, q3])
    return idxs[:16]

def make_video_transform():
    return T.Compose([
        T.Resize((FRAME_SIZE, FRAME_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomGrayscale(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]),
    ])

def get_16_frames(vr: VideoReader, start_f: int, end_f: int, v_transform):
    idxs = sample_16_quartile(start_f, end_f)
    frames = vr.get_batch(idxs).asnumpy()  # [16,H,W,3]
    out = []
    for f in frames:
        img = Image.fromarray(f)
        out.append(v_transform(img))
    return torch.stack(out, dim=0)  # [16,3,224,224]

# ---------------------------
# identity별 mp4 하나만 고르는 샘플 수집
# ---------------------------
def collect_one_video_per_identity(version_root: Path) -> List[Path]:
    mp4s = [p for p in version_root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    by_id: Dict[Path, List[Path]] = {}
    for p in mp4s:
        id_dir = p.parent  # .../race/gender/identity
        by_id.setdefault(id_dir, []).append(p)

    picked = []
    for id_dir, vids in by_id.items():
        picked.append(random.choice(vids))
    return picked

# ---------------------------
# Process per video (save separate .npy)
# ---------------------------
def process_one_video(
    video_path: Path,
    label: int,
    version_name: str,
    out_root: Path,
    mel_transform, v_transform,
    clips_per_video: int,
):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    total_len = len(vr)
    wav = wav_from_video(video_path)

    version_dirname = VERSION_DIR[version_name]
    parts = video_path.parts
    vidx = parts.index(version_dirname)
    rel_under_version = Path(*parts[vidx+1:])  # race/gender/id/xxx.mp4

    for c in range(clips_per_video):
        start_f, end_f = sample_clip_indices(total_len, fps)
        start_sec = start_f / fps

        v_clip = get_16_frames(vr, start_f, end_f, v_transform)  # [16,3,224,224]
        mel_z = lrs3_mel_clip(wav, start_sec, mel_transform)     # [128,768]

        key = f"{video_path.stem}_clip{c:02d}"
        rel_parent = rel_under_version.parent  # race/gender/id

        # 저장 루트: out_root/{mel,v_clip,label}/{version_dir}/race/gender/id/key.npy
        def sp(subroot: str):
            d = out_root / subroot / version_dirname / rel_parent
            d.mkdir(parents=True, exist_ok=True)
            return d / f"{key}.npy"

        np.save(sp("mel"),    mel_z.numpy().astype(np.float32))
        np.save(sp("v_clip"), v_clip.numpy().astype(np.float32))
        np.save(sp("label"),  np.float32(label))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--versions", type=str, default="ALL")
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--clips_per_video", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    if args.versions.strip().upper() == "ALL":
        versions = ALL_VERSIONS
    else:
        versions = [s.strip() for s in args.versions.split(",") if s.strip()]
    print(f"[Preproc] Versions: {versions}")

    samples: List[Tuple[Path,int,str]] = []
    for ver in versions:
        vroot = data_root / VERSION_DIR[ver]
        if not vroot.exists():
            print(f"[WARN] missing version dir: {vroot}")
            continue
        label = 0 if ver == "RVRA" else 1
        mp4s = collect_one_video_per_identity(vroot)
        samples.extend([(m, label, ver) for m in mp4s])

    print(f"[Preproc] Total mp4s to process (1 per identity per version): {len(samples)}")

    mel_transform = make_lrs3_mel_transform()
    v_transform = make_video_transform()

    for (vpath, label, ver) in tqdm(samples, desc="Preprocessing", ncols=100):
        try:
            process_one_video(
                vpath, label, ver, out_root,
                mel_transform, v_transform,
                args.clips_per_video
            )
        except Exception as e:
            print(f"\n[WARN] failed: {vpath} ({e})")

    print("Preprocessing finished.")

if __name__ == "__main__":
    main()
