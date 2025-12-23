# baseline.py
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio

# -------------------------
# 경로 설정: OpenAVFF/src 추가
# -------------------------
ROOT = Path(__file__).resolve().parent
OPENAVFF_SRC = ROOT / "OpenAVFF" / "src"
sys.path.append(str(OPENAVFF_SRC))

# OpenAVFF 쪽 모듈 (모델 / 학습 루프 그대로 사용)
from models.video_cav_mae import VideoCAVMAEFT
from traintest_ft import train

# 우리 쪽 데이터스플릿/버전 정의 재사용
from src.dataset import read_identities, build_key_list_dual, VERSION_DIR


# ============================================================
# Dataset: preproc_root 기반 FakeAVCeleb AV baseline용
# ============================================================
class FakeAVCelebBaselineTrain(Dataset):
    """
    preproc_root 아래의 mel / mel_err, v_clip npy를 이용해
    OpenAVFF baseline(VideoCAVMAEFT)이 기대하는 형태로 반환하는 학습용 Dataset.

    return:
      audio_fbank: [T, 128]  (기본 T=1024)
      video:       [3, 16, 224, 224]
      label:       [1 - cls, cls]  (0=real, 1=fake)
    """
    def __init__(
        self,
        preproc_root: str,
        items,
        audio_subdir: str = "mel_err",
        target_length: int = 1024,
        num_frames: int = 16,
        mean: float = 0.0,
        std: float = 1.0,
        freqm: int = 0,
        timem: int = 0,
        noise: bool = False,
    ):
        self.preproc_root = Path(preproc_root)
        self.items = list(items)
        self.audio_subdir = audio_subdir
        self.target_length = target_length
        self.num_frames = num_frames

        self.mean = float(mean)
        self.std = float(std if std > 0 else 1.0)

        self.freqm = freqm
        self.timem = timem
        self.noise = noise

        self.freq_mask = torchaudio.transforms.FrequencyMasking(self.freqm) if self.freqm > 0 else None
        self.time_mask = torchaudio.transforms.TimeMasking(self.timem) if self.timem > 0 else None

    def __len__(self):
        return len(self.items)

    def _load_npy(self, subdir: str, relkey: str) -> torch.Tensor:
        p = self.preproc_root / subdir / relkey
        arr = np.load(p, mmap_mode="r").astype(np.float32)
        return torch.from_numpy(np.array(arr, copy=False))

    def _load_audio_fbank(self, relkey: str) -> torch.Tensor:
        """
        mel / mel_err npy를 읽어서 baseline이 기대하는 [T, 128] 형태로 변환.
        - 원본: [F(=128), T0] 형태라고 가정
        - 시간축을 target_length 로 선형 보간
        """
        x = self._load_npy(self.audio_subdir, relkey)  # [F, T0] 또는 [T0, F]
        if x.ndim != 2:
            raise RuntimeError(f"Audio npy must be 2D, got {x.shape} for {relkey}")

        # (가능한 한) [F, T0] → [T0, F] 로 맞추기
        # 우리의 mel_err는 [128, T] 형태이므로 그대로 transpose
        if x.shape[0] == 128:  # [F, T0]
            x = x.transpose(0, 1)  # [T0, F]
        elif x.shape[1] == 128:  # [T0, F]
            pass
        else:
            # 둘 다 128이 아니면 에러
            raise RuntimeError(f"Expected one dim to be 128, got {x.shape} for {relkey}")

        T0, F_ = x.shape
        if F_ != 128:
            raise RuntimeError(f"Expected 128 mel bins, got {F_} for {relkey}")

        # 시간축을 target_length로 보간
        if T0 != self.target_length:
            x_t = x.transpose(0, 1).unsqueeze(0)  # [1, F, T0]
            x_t = F.interpolate(
                x_t, size=self.target_length, mode="linear", align_corners=False
            )  # [1, F, target_length]
            x = x_t.squeeze(0).transpose(0, 1)  # [target_length, F]

        fbank = x  # [T, 128]

        # SpecAug (train만)
        if self.freq_mask is not None or self.time_mask is not None:
            fb = fbank.transpose(0, 1).unsqueeze(0)  # [1, F, T]
            if self.freq_mask is not None:
                fb = self.freq_mask(fb)
            if self.time_mask is not None:
                fb = self.time_mask(fb)
            fbank = fb.squeeze(0).transpose(0, 1)  # [T, F]

        # 정규화
        fbank = (fbank - self.mean) / self.std

        # 노이즈
        if self.noise:
            noise = torch.rand_like(fbank) * (np.random.rand() / 10.0)
            shift = np.random.randint(-self.target_length, self.target_length)
            fbank = fbank + noise
            fbank = torch.roll(fbank, shifts=shift, dims=0)

        return fbank  # [T, 128]

    def _load_video_clip(self, relkey: str) -> torch.Tensor:
        """
        v_clip npy를 읽어서 [3, num_frames, 224, 224] 로 변환.
        - 저장 형식은 [T, C, H, W] 라고 가정
        """
        v = self._load_npy("v_clip", relkey).float()  # [T, C, H, W]
        if v.ndim != 4:
            raise RuntimeError(f"v_clip must be 4D, got {v.shape} for {relkey}")

        T, C, H, W = v.shape
        if T != self.num_frames:
            # 길이가 다를 경우 균일 샘플링
            idx = np.linspace(0, T - 1, self.num_frames).astype(int)
            v = v[idx]

        # [T, C, H, W] -> [C, T, H, W]
        v = v.permute(1, 0, 2, 3).contiguous()
        return v

    def __getitem__(self, idx):
        relkey, y_video, y_audio, ver = self.items[idx]

        # audio
        fbank = self._load_audio_fbank(relkey)  # [T, 128]

        # video
        frames = self._load_video_clip(relkey)  # [3, T, H, W]

        # label: video fake 여부 기준 (0=real, 1=fake)
        cls = int(y_video)
        label = torch.tensor([1 - cls, cls], dtype=torch.float32)

        return fbank, frames, label


class FakeAVCelebBaselineEval(Dataset):
    """
    검증용 Dataset.
    train과 동일하지만 SpecAug/노이즈 없음, misclassified용 path 문자열을 함께 반환.

    return:
      audio_fbank: [T, 128]
      video:       [3, 16, 224, 224]
      label:       [1 - cls, cls]
      path_str:    str  (예: mel_err/relkey 또는 v_clip/relkey)
    """
    def __init__(
        self,
        preproc_root: str,
        items,
        audio_subdir: str = "mel_err",
        target_length: int = 1024,
        num_frames: int = 16,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        self.preproc_root = Path(preproc_root)
        self.items = list(items)
        self.audio_subdir = audio_subdir
        self.target_length = target_length
        self.num_frames = num_frames

        self.mean = float(mean)
        self.std = float(std if std > 0 else 1.0)

    def __len__(self):
        return len(self.items)

    def _load_npy(self, subdir: str, relkey: str) -> torch.Tensor:
        p = self.preproc_root / subdir / relkey
        arr = np.load(p, mmap_mode="r").astype(np.float32)
        return torch.from_numpy(np.array(arr, copy=False))

    def _load_audio_fbank(self, relkey: str) -> torch.Tensor:
        x = self._load_npy(self.audio_subdir, relkey)
        if x.ndim != 2:
            raise RuntimeError(f"Audio npy must be 2D, got {x.shape} for {relkey}")

        if x.shape[0] == 128:
            x = x.transpose(0, 1)  # [T0, F]
        elif x.shape[1] == 128:
            pass
        else:
            raise RuntimeError(f"Expected one dim to be 128, got {x.shape} for {relkey}")

        T0, F_ = x.shape
        if F_ != 128:
            raise RuntimeError(f"Expected 128 mel bins, got {F_} for {relkey}")

        if T0 != self.target_length:
            x_t = x.transpose(0, 1).unsqueeze(0)  # [1, F, T0]
            x_t = F.interpolate(
                x_t, size=self.target_length, mode="linear", align_corners=False
            )
            x = x_t.squeeze(0).transpose(0, 1)  # [T, F]

        fbank = x
        fbank = (fbank - self.mean) / self.std
        return fbank

    def _load_video_clip(self, relkey: str) -> torch.Tensor:
        v = self._load_npy("v_clip", relkey).float()  # [T, C, H, W]
        if v.ndim != 4:
            raise RuntimeError(f"v_clip must be 4D, got {v.shape} for {relkey}")

        T, C, H, W = v.shape
        if T != self.num_frames:
            idx = np.linspace(0, T - 1, self.num_frames).astype(int)
            v = v[idx]

        v = v.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        return v

    def __getitem__(self, idx):
        relkey, y_video, y_audio, ver = self.items[idx]

        fbank = self._load_audio_fbank(relkey)
        frames = self._load_video_clip(relkey)

        cls = int(y_video)
        label = torch.tensor([1 - cls, cls], dtype=torch.float32)

        # misclassified 리스트에 찍힐 path 문자열 (audio 기준 경로로 통일)
        path_str = str(self.preproc_root / self.audio_subdir / relkey)

        return fbank, frames, label, path_str


# ============================================================
# Audio mean / std 자동 계산 (train items 기준)
# ============================================================
def compute_audio_stats(preproc_root: str, items, audio_subdir: str, target_length: int = 1024):
    preproc_root = Path(preproc_root)

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for relkey, y_video, y_audio, ver in items:
        p = preproc_root / audio_subdir / relkey
        arr = np.load(p, mmap_mode="r").astype(np.float32)
        x = torch.from_numpy(np.array(arr, copy=False))

        if x.ndim != 2:
            continue

        if x.shape[0] == 128:
            x = x.transpose(0, 1)  # [T0, F]
        elif x.shape[1] == 128:
            pass
        else:
            continue

        T0, F_ = x.shape
        if F_ != 128:
            continue

        if T0 != target_length:
            x_t = x.transpose(0, 1).unsqueeze(0)  # [1, F, T0]
            x_t = F.interpolate(
                x_t, size=target_length, mode="linear", align_corners=False
            )
            x = x_t.squeeze(0).transpose(0, 1)  # [T, F]

        flat = x.reshape(-1)
        total_sum += flat.sum().item()
        total_sq_sum += (flat ** 2).sum().item()
        total_count += flat.numel()

    if total_count == 0:
        raise RuntimeError("No valid audio found for computing stats.")

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean ** 2
    std = float(np.sqrt(max(var, 1e-8)))

    return float(mean), std


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="OpenAVFF baseline on FakeAVCeleb preproc (mel / mel_err + v_clip)")

    # 데이터 루트 / split
    p.add_argument("--preproc_root", type=str, required=True)
    p.add_argument("--split_dir", type=str, default="data/splits")
    p.add_argument("--train_split", type=str, required=True)
    p.add_argument("--val_split", type=str, required=True)

    # 버전 구성: real + fake 리스트
    p.add_argument("--real_version", type=str, default="RVRA")
    p.add_argument("--train_fakes", type=str, default="RVFA,RVFA-VC,RVFA-SVC")
    p.add_argument("--val_fakes", type=str, default="RVFA,RVFA-VC,RVFA-SVC")

    # audio 소스: mel_err / mel 중 선택
    p.add_argument("--audio_subdir", type=str, default="mel_err",
                   help="preproc_root 아래 audio 서브디렉토리 이름 (예: mel_err, mel)")

    # 하이퍼파라미터 (OpenAVFF run_ft와 최대한 호환)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument("--metrics", type=str, default="mAP", choices=["mAP", "acc"])
    p.add_argument("--loss", type=str, default="CE", choices=["BCE", "CE"])
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--n_classes', type=int, default=2)

    p.add_argument("--lrscheduler_start", type=int, default=10)
    p.add_argument("--lrscheduler_step", type=int, default=5)
    p.add_argument("--lrscheduler_decay", type=float, default=0.5)

    p.add_argument('--head_lr', type=float, default=10.0)

    # audio 관련
    p.add_argument('--target_length', type=int, default=1024)
    p.add_argument('--freqm', type=int, default=0)
    p.add_argument('--timem', type=int, default=0)
    p.add_argument('--noise', type=bool, default=False)

    # pretrained checkpoint
    p.add_argument('--pretrain_path', type=str, default=None)

    args = p.parse_args()

    preproc_root = Path(args.preproc_root)
    split_dir = Path(args.split_dir)

    # -----------------------------
    # split 읽기 (identity level)
    # -----------------------------
    train_ids = read_identities(str(split_dir / args.train_split))
    val_ids = read_identities(str(split_dir / args.val_split))

    train_fakes = [s.strip() for s in args.train_fakes.split(",") if s.strip()]
    val_fakes = [s.strip() for s in args.val_fakes.split(",") if s.strip()]

    # -----------------------------
    # item 리스트 구성
    # (src.dataset.build_key_list_dual 재사용)
    # -----------------------------
    train_items = build_key_list_dual(
        args.preproc_root,
        train_ids,
        real_version=args.real_version,
        fake_versions=train_fakes,
    )
    val_items = build_key_list_dual(
        args.preproc_root,
        val_ids,
        real_version=args.real_version,
        fake_versions=val_fakes,
    )

    print(f"[Baseline] #train items = {len(train_items)}, #val items = {len(val_items)}")

    # -----------------------------
    # audio mean / std 자동 계산
    # -----------------------------
    print("[Baseline] Estimating dataset audio mean/std from train items...")
    mean, std = compute_audio_stats(
        args.preproc_root,
        train_items,
        audio_subdir=args.audio_subdir,
        target_length=args.target_length,
    )
    print(f"[Baseline] Audio mean={mean:.4f}, std={std:.4f}")

    # -----------------------------
    # Dataset / DataLoader 구성
    # -----------------------------
    train_ds = FakeAVCelebBaselineTrain(
        args.preproc_root,
        train_items,
        audio_subdir=args.audio_subdir,
        target_length=args.target_length,
        num_frames=16,
        mean=mean,
        std=std,
        freqm=args.freqm,
        timem=args.timem,
        noise=args.noise,
    )
    val_ds = FakeAVCelebBaselineEval(
        args.preproc_root,
        val_items,
        audio_subdir=args.audio_subdir,
        target_length=args.target_length,
        num_frames=16,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"[Baseline] Using Train batches={len(train_loader)}, Val batches={len(val_loader)}")

    # -----------------------------
    # 모델 생성 + (옵션) pretrained 로드
    # -----------------------------
    cavmae_ft = VideoCAVMAEFT(n_classes=args.n_classes)

    if args.pretrain_path is not None:
        print(f"[Baseline] Loading pretrained weights from {args.pretrain_path}")
        mdl_weight = torch.load(args.pretrain_path, map_location="cpu")
        if not isinstance(cavmae_ft, torch.nn.DataParallel):
            cavmae_ft = torch.nn.DataParallel(cavmae_ft)
        missing, unexpected = cavmae_ft.load_state_dict(mdl_weight, strict=False)
        print("Missing keys: ", len(missing))
        print("Unexpected keys: ", len(unexpected))
        if len(missing) > 0:
            print("  Missing:", missing[:10], "...")
        if len(unexpected) > 0:
            print("  Unexpected:", unexpected[:10], "...")
    else:
        print("[Baseline] WARNING: Training baseline without pretrain weights.")

    # -----------------------------
    # 학습 시작 (traintest_ft.train 그대로 사용)
    # -----------------------------
    print(f"[Baseline] Start training for {args.n_epochs} epochs")
    train(cavmae_ft, train_loader, val_loader, args)


if __name__ == "__main__":
    main()
