#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

VALID_CATS = ("RVFA", "FVRA", "FVFA")
REAL_PREFIX = "/data2/local_datasets/jhlee39/FakeAVCeleb_v1.2"

def find_single_mp4(dir_path: Path) -> Optional[Path]:
    exts = [".mp4", ".mkv", ".mov", ".webm"]
    files = [p for p in dir_path.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    if not files:
        return None
    files.sort()
    return files[0]

def list_all_mp4s(dir_path: Path) -> List[Path]:
    exts = [".mp4", ".mkv", ".mov", ".webm"]
    files = [p for p in dir_path.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    files.sort()
    return files

def build_identity_index(data_root: Path, vc_mode: str, only_rvfa_build: bool) -> Dict[str, dict]:
    RVRA = data_root / "RealVideo-RealAudio"
    RVFA_base = data_root / "RealVideo-FakeAudio-Refine"
    if vc_mode == "hardest":
        RVFA = RVFA_base / "hardest"
    elif vc_mode == "harder":
        RVFA = RVFA_base / "harder"
    else:
        RVFA = data_root / "RealVideo-FakeAudio"

    FVRA = data_root / "FakeVideo-RealAudio"
    FVFA = data_root / "FakeVideo-FakeAudio"

    if not RVRA.is_dir():
        raise FileNotFoundError(f"Missing directory: {RVRA}")

    identities = {}
    for race_dir in sorted(RVRA.iterdir()):
        for gender_dir in sorted(race_dir.iterdir()):
            for id_dir in sorted(gender_dir.iterdir()):
                identity = id_dir.name
                key = f"{race_dir.name}/{gender_dir.name}/{identity}"
                real_path = find_single_mp4(id_dir)
                if real_path is None:
                    continue
                identities[key] = {
                    "race": race_dir.name,
                    "gender": gender_dir.name,
                    "real": real_path.resolve(),
                    "fakes": {"RVFA": [], "FVRA": [], "FVFA": []},
                }

    def attach_fakes(root_path: Path, bucket_name: str):
        if not root_path.is_dir():
            return
        for race_dir in sorted(root_path.iterdir()):
            for gender_dir in sorted(race_dir.iterdir()):
                for id_dir in sorted(gender_dir.iterdir()):
                    key = f"{race_dir.name}/{gender_dir.name}/{id_dir.name}"
                    if key not in identities:
                        continue
                    files = list_all_mp4s(id_dir)
                    if files:
                        identities[key]["fakes"][bucket_name].extend([p.resolve() for p in files])

    if only_rvfa_build:
        attach_fakes(RVFA, "RVFA")
    else:
        attach_fakes(RVFA, "RVFA")
        attach_fakes(FVRA, "FVRA")
        attach_fakes(FVFA, "FVFA")

    filtered = {k: v for k, v in identities.items() if sum(len(v["fakes"][cat]) for cat in VALID_CATS) > 0}
    return filtered

def stratified_identity_split(identities: Dict[str, dict], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for k, item in identities.items():
        buckets[(item["race"], item["gender"])].append(k)

    print("[INFO] Buckets (race,gender -> #ids):")
    for (race, gender), keys in sorted(buckets.items()):
        print(f"  - ({race},{gender}): {len(keys)}")

    train_keys, test_keys = [], []
    for (race, gender), keys in buckets.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_train = round(train_ratio * len(keys))
        if n_train == 0 and len(keys) > 1:
            n_train = 1
        if n_train == len(keys) and len(keys) > 1:
            n_train = len(keys) - 1
        train_keys.extend(keys[:n_train])
        test_keys.extend(keys[n_train:])
    return train_keys, test_keys

def hash_based_perm(key: str, seed: int, cats: List[str]) -> List[str]:
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    idxs = list(range(len(cats)))
    rnd = random.Random(h)
    rnd.shuffle(idxs)
    return [cats[i] for i in idxs]

def choose_fakes_for_identity(item: dict, k: int, seed: int, key: str,
                              only_rvfa: bool, phase: str,
                              cross_manip: Optional[str]) -> List[Path]:
    f = item["fakes"]

    if cross_manip:
        active_cats = [cross_manip] if phase == "test" else [c for c in VALID_CATS if c != cross_manip]
    else:
        active_cats = ["RVFA"] if only_rvfa else list(VALID_CATS)

    active_cats = [c for c in active_cats if len(f[c]) > 0]
    if not active_cats:
        return []

    order = hash_based_perm(key, seed, active_cats)
    pools = {name: sorted(f[name], key=lambda p: str(p)) for name in order}

    out = []
    idx_round = 0
    while len(out) < k:
        picked_any = False
        for cat in order:
            pool = pools[cat]
            if not pool:
                continue
            pick = pool[idx_round % len(pool)]
            out.append(pick)
            picked_any = True
            if len(out) >= k:
                break
        if not picked_any:
            break
        idx_round += 1
    while len(out) < k:
        pool = pools[order[0]]
        if not pool:
            break
        out.append(pool[len(out) % len(pool)])
    return out

def write_csv(save_path: Path, rows: List[Tuple[Path, int]]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "target"])
        for p, lab in rows:
            abs_path = Path(str(p).replace(
                "/data/jhlee39/workspace/repos/AVDFD/data/FakeAVCeleb_Refine/FakeAVCeleb_v1.2",
                "/data2/local_datasets/jhlee39/FakeAVCeleb_Refine/FakeAVCeleb_v1.2"
            ))
            writer.writerow([str(abs_path), lab])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--fake-mult", type=int, default=3)
    ap.add_argument("--only-RVFA", action="store_true")
    ap.add_argument("--vc", type=str, default="false", choices=["false", "harder", "hardest"])
    ap.add_argument("--cross-manip", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    only_rvfa_build = args.only_RVFA and args.cross_manip == ""
    vc_mode = args.vc
    cross_manip = args.cross_manip.strip().upper() or None

    print(f"[INFO] Using vc_mode = {vc_mode}")
    identities = build_identity_index(data_root, vc_mode=vc_mode, only_rvfa_build=only_rvfa_build)
    train_ids, test_ids = stratified_identity_split(identities, train_ratio=args.train_ratio, seed=args.seed)

    def build_rows(id_list, phase):
        rows = []
        for key in id_list:
            item = identities[key]
            rows.append((item["real"], 0))
            k = 1 if only_rvfa_build else args.fake_mult
            fakes = choose_fakes_for_identity(item, k, args.seed, key, only_rvfa_build, phase, cross_manip)
            for fp in fakes:
                rows.append((fp, 1))
        return rows

    train_rows = build_rows(train_ids, "train")
    test_rows = build_rows(test_ids, "test")

    write_csv(out_dir / "trainset.csv", train_rows)
    write_csv(out_dir / "testset.csv", test_rows)

    print("=== Done ===")
    print(f"Train: real={sum(1 for _,l in train_rows if l==0)}, fake={sum(1 for _,l in train_rows if l==1)}")
    print(f"Test : real={sum(1 for _,l in test_rows if l==0)}, fake={sum(1 for _,l in test_rows if l==1)}")

if __name__ == "__main__":
    main()
