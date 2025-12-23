# src/dataset.py
from pathlib import Path
from typing import List, Tuple, Dict
import random

import numpy as np
import torch
from torch.utils.data import Dataset

VERSION_DIR = {
    "RVRA": "RealVideo-RealAudio",
    "RVFA": "RealVideo-FakeAudio",
    "FVRA": "FakeVideo-RealAudio",
    "FVFA": "FakeVideo-FakeAudio",
    "RVFA-VC": "RealVideo-FakeAudio-VoiceClone",
    "RVFA-SVC": "RealVideo-FakeAudio-SelfVoiceClone",
}

# audio 조작 여부 라벨 (pretrain aux / main audio head)
AUDIO_LABEL = {
    "RVRA": 0,
    "FVRA": 0,
    "RVFA": 1,
    "FVFA": 1,
    "RVFA-VC": 1,
    "RVFA-SVC": 1,
}

# pretrain용 4-class 라벨
# (ordinal 구조: RVRA(0) < RVFA-SVC(1) < RVFA-VC(2) < RVFA(3))
PRETRAIN_CLASS = {
    "RVRA": 0,
    "RVFA-SVC": 1,
    "RVFA-VC": 2,
    "RVFA": 3,
}


def read_identities(split_path: str) -> List[str]:
    ids = []
    with open(split_path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def find_keys_for_id(mel_err_version_root: Path, identity: str) -> List[Path]:
    hits = []
    for p in mel_err_version_root.rglob("*.npy"):
        if p.parent.name == identity:
            hits.append(p)
    return hits


def build_key_list_dual(
    preproc_root: str,
    identities: List[str],
    real_version: str,
    fake_versions: List[str],
) -> List[Tuple[str, int, int, str]]:
    """
    main dual 학습용 item 만들기
    return: [(relkey, video_label, audio_label, version)]
    relkey = vdir/race/gender/id/key.npy  (audio/visual 공통 key)
    """
    preproc_root = Path(preproc_root)
    items: List[Tuple[str, int, int, str]] = []

    def add_version(ver: str, v_lab: int):
        vdir = VERSION_DIR[ver]
        # key 탐색은 mel_err 기준으로 하지만,
        # mel, v_clip 모두 동일한 relkey 구조를 공유
        mel_err_root = preproc_root / "mel_err" / vdir
        a_lab = AUDIO_LABEL[ver]
        for identity in identities:
            keys = find_keys_for_id(mel_err_root, identity)
            for k in keys:
                rel = k.relative_to(preproc_root / "mel_err")
                items.append((str(rel), v_lab, a_lab, ver))

    add_version(real_version, 0)
    for fv in fake_versions:
        add_version(fv, 1)

    return items


def build_fake_relkeys_for_pretrain(
    preproc_root: str,
    identities: List[str],
    fake_versions: List[str],
    audio_subdir: str = "mel_err",
) -> List[Tuple[str, str]]:
    """
    pretrain pair용 fake relkey 목록
    return: [(fake_relkey, fake_version)]
    fake_relkey = vdir/race/gender/id/key.npy (audio_subdir 기준)
    audio_subdir: "mel_err" (DIRE) 또는 "mel"
    """
    preproc_root = Path(preproc_root)
    out: List[Tuple[str, str]] = []

    for fv in fake_versions:
        vdir = VERSION_DIR[fv]
        mel_root = preproc_root / audio_subdir / vdir
        for identity in identities:
            keys = find_keys_for_id(mel_root, identity)
            for k in keys:
                rel = k.relative_to(preproc_root / audio_subdir)
                out.append((str(rel), fv))

    return out


class FakeAVCelebPretrainPairs(Dataset):
    """
    pretrain용 pair dataset.
    각 fake sample에 대해 동일 id 폴더의 RVRA(real) mel_err 또는 mel 1개를 찾아 반환.

    returns:
      mel_real, mel_fake, cls_label(0~3), bin_label(0/1), relkey_fake
    """
    def __init__(
        self,
        preproc_root: str,
        fake_relkeys: List[Tuple[str, str]],
        seed: int = 0,
        audio_subdir: str = "mel_err",
    ):
        self.preproc_root = Path(preproc_root)
        self.fake_relkeys = fake_relkeys
        self.audio_subdir = audio_subdir  # "mel_err" or "mel"
        random.seed(seed)

        self.real_key_by_iddir: Dict[str, str] = {}
        rvra_dir = VERSION_DIR["RVRA"]
        mel_rvra_root = self.preproc_root / self.audio_subdir / rvra_dir

        for id_dir in mel_rvra_root.rglob("*"):
            if id_dir.is_dir():
                npys = sorted(list(id_dir.glob("*.npy")))
                if len(npys) == 0:
                    continue
                rel = npys[0].relative_to(self.preproc_root / self.audio_subdir)
                key = str(id_dir.relative_to(mel_rvra_root))  # race/gender/id
                self.real_key_by_iddir[key] = str(rel)

    def __len__(self):
        return len(self.fake_relkeys)

    def _load_npy(self, relkey: str):
        p = self.preproc_root / self.audio_subdir / relkey
        arr = np.load(p, mmap_mode="r").astype(np.float32)
        return torch.from_numpy(np.array(arr, copy=False))

    def _get_id_folder_key(self, relkey: str) -> str:
        parts = Path(relkey).parts
        return str(Path(*parts[1:4]))  # race/gender/id

    def __getitem__(self, idx):
        fake_relkey, fake_ver = self.fake_relkeys[idx]
        id_folder_key = self._get_id_folder_key(fake_relkey)

        if id_folder_key not in self.real_key_by_iddir:
            raise FileNotFoundError(
                f"[PretrainPairs] RVRA real npy not found for id folder: {id_folder_key}"
            )

        real_relkey = self.real_key_by_iddir[id_folder_key]

        mel_fake = self._load_npy(fake_relkey).float()
        mel_real = self._load_npy(real_relkey).float()

        cls_lab = PRETRAIN_CLASS[fake_ver]
        bin_lab = AUDIO_LABEL[fake_ver]

        return (
            mel_real,
            mel_fake,
            torch.tensor(cls_lab, dtype=torch.long),
            torch.tensor(bin_lab, dtype=torch.float32),
            fake_relkey,
        )


class FakeAVCelebDual(Dataset):
    """
    main dual 학습용 dataset.
    returns:
      mel, v_clip, first_frame, y_video, y_audio, version, relkey
    """
    def __init__(self, preproc_root: str, items: List[Tuple[str,int,int,str]], seed=0):
        self.preproc_root = Path(preproc_root)
        self.items = items
        random.seed(seed)

    def __len__(self):
        return len(self.items)

    def _load_npy(self, subdir: str, relkey: str):
        p = self.preproc_root / subdir / relkey
        arr = np.load(p, mmap_mode="r").astype(np.float32)
        return torch.from_numpy(np.array(arr, copy=False))

    def __getitem__(self, idx):
        relkey, y_video, y_audio, ver = self.items[idx]

        mel = self._load_npy("mel", relkey).float()       # [128,768]
        v_clip = self._load_npy("v_clip", relkey).float() # [16,3,224,224]
        first_frame = v_clip[0]

        return (
            mel,
            v_clip,
            first_frame,
            torch.tensor(y_video, dtype=torch.float32),
            torch.tensor(y_audio, dtype=torch.float32),
            ver,
            relkey,
        )
