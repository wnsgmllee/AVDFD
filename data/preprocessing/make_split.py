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

def build_identity_index(data_root: Path, vc_mode: bool, only_rvfa_build: bool) -> Dict[str, dict]:
    RVRA = data_root / "RealVideo-RealAudio"
    RVFA_name = "RealVideo-FakeAudio-Refine" if vc_mode else "RealVideo-FakeAudio"
    RVFA = data_root / RVFA_name
    FVRA = data_root / "FakeVideo-RealAudio"
    FVFA = data_root / "FakeVideo-FakeAudio"

    if not RVRA.is_dir():
        raise FileNotFoundError(f"Missing directory: {RVRA}")

    identities = {}
    # RVRA (race/gender/identity)
    for race_dir in sorted([d for d in RVRA.iterdir() if d.is_dir()]):
        race = race_dir.name
        for gender_dir in sorted([d for d in race_dir.iterdir() if d.is_dir()]):
            gender = gender_dir.name
            for id_dir in sorted([d for d in gender_dir.iterdir() if d.is_dir()]):
                identity = id_dir.name
                key = f"{race}/{gender}/{identity}"
                real_path = find_single_mp4(id_dir)
                if real_path is None:
                    continue
                identities[key] = {
                    "race": race,
                    "gender": gender,
                    "real": real_path.resolve(),
                    "fakes": {"RVFA": [], "FVRA": [], "FVFA": []},
                }

    def attach_fakes(root_path: Path, bucket_name: str):
        if not root_path.is_dir():
            return
        for race_dir in sorted([d for d in root_path.iterdir() if d.is_dir()]):
            race = race_dir.name
            for gender_dir in sorted([d for d in race_dir.iterdir() if d.is_dir()]):
                gender = gender_dir.name
                for id_dir in sorted([d for d in gender_dir.iterdir() if d.is_dir()]):
                    identity = id_dir.name
                    key = f"{race}/{gender}/{identity}"
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

    # 최소 1개 fake 있는 identity만 사용
    filtered = {}
    for k, item in identities.items():
        total_fake = sum(len(v) for v in item["fakes"].values())
        if total_fake > 0:
            filtered[k] = item
    return filtered

def stratified_identity_split(identities: Dict[str, dict], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for k, item in identities.items():
        buckets[(item["race"], item["gender"])].append(k)

    # 로그: 버킷 분포
    print("[INFO] Buckets (race,gender -> #ids):")
    for (race, gender), keys in sorted(buckets.items()):
        print(f"  - ({race},{gender}): {len(keys)}")

    train_keys, test_keys = [], []
    for (race, gender), keys in buckets.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = round(train_ratio * n)
        if n_train == 0 and n > 1:
            n_train = 1
        if n_train == n and n > 1:
            n_train = n - 1
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
                              only_rvfa: bool,
                              phase: str,  # "train" or "test"
                              cross_manip: Optional[str]) -> List[Path]:
    f = item["fakes"]

    if cross_manip:
        if cross_manip not in VALID_CATS:
            raise ValueError(f"--cross-manip must be one of {VALID_CATS}, got {cross_manip}")
        if phase == "test":
            active_cats = [cross_manip]
        else:
            active_cats = [c for c in VALID_CATS if c != cross_manip]
    else:
        if only_rvfa:
            active_cats = ["RVFA"]
        else:
            active_cats = list(VALID_CATS)

    # 실제로 존재하는 카테고리만
    active_cats = [c for c in active_cats if len(f[c]) > 0]
    if not active_cats:
        return []

    order = hash_based_perm(key, seed, active_cats)
    pools = {name: sorted(f[name], key=lambda p: str(p)) for name in order}

    out: List[Path] = []
    idx_round = 0
    while len(out) < k:
        picked_any = False
        for cat in order:
            pool = pools[cat]
            if not pool:
                continue
            pick = pool[(idx_round) % len(pool)]
            out.append(pick)
            picked_any = True
            if len(out) >= k:
                break
        if not picked_any:
            break
        idx_round += 1

    while len(out) < k:
        base_cat = order[0]
        pool = pools[base_cat]
        if not pool:
            break
        out.append(pool[len(out) % len(pool)])
    return out

def write_csv(save_path: Path, rows: List[Tuple[Path, int]]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "target"])
        for p, lab in rows:
            w.writerow([str(Path(p).resolve()), lab])

def main():
    ap = argparse.ArgumentParser(description="FakeAVCeleb split CSV generator (ID-disjoint, stratified; improved sampling & cross-manip; verbose).")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--fake-mult", type=int, default=3, help="fake = real x K (ignored if --only-RVFA & no cross-manip)")
    ap.add_argument("--only-RVFA", action="store_true", help="Use only RVFA for fakes (ignored if --cross-manip is set).")
    ap.add_argument("--vc", action="store_true", help="Use RealVideo-FakeAudio-Refine instead of RealVideo-FakeAudio.")
    ap.add_argument("--cross-manip", type=str, default="", help='One of {"RVFA","FVRA","FVFA"}. If set: TRAIN excludes it; TEST uses only it.')
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    fake_mult = max(1, args.fake_mult)
    only_rvfa_flag = args.only_RVFA
    vc_mode = args.vc
    cross_manip = args.cross_manip.strip().upper()
    if cross_manip == "":
        cross_manip = None

    # cross-manip이 설정되면 only_rvfa는 무시
    only_rvfa_build = (only_rvfa_flag and (cross_manip is None))

    print("[INFO] data_root =", data_root)
    print("[INFO] out_dir   =", out_dir)
    print("[INFO] train_ratio =", args.train_ratio)
    print("[INFO] fake_mult   =", fake_mult, "(ignored if only_RVFA & no cross-manip)")
    print("[INFO] only_RVFA   =", only_rvfa_flag, "(build-only:", only_rvfa_build, ")")
    print("[INFO] vc_mode     =", vc_mode)
    print("[INFO] cross_manip =", cross_manip)
    print("[INFO] seed        =", args.seed)

    identities = build_identity_index(data_root, vc_mode=vc_mode, only_rvfa_build=only_rvfa_build)
    print(f"[INFO] identities with >=1 fake: {len(identities)}")

    # 카테고리 분포 로그
    total_real = 0
    cat_cnt = {"RVFA":0, "FVRA":0, "FVFA":0}
    for item in identities.values():
        total_real += 1
        for c in VALID_CATS:
            cat_cnt[c] += len(item["fakes"][c])
    print(f"[INFO] per-category fake files: RVFA={cat_cnt['RVFA']}, FVRA={cat_cnt['FVRA']}, FVFA={cat_cnt['FVFA']}")

    if not identities:
        raise RuntimeError("No valid identities (need RVRA + at least one fake). Check folder names and mp4 presence.")

    train_ids, test_ids = stratified_identity_split(identities, train_ratio=args.train_ratio, seed=args.seed)
    print(f"[INFO] split -> train IDs: {len(train_ids)}, test IDs: {len(test_ids)}")

    def build_rows(id_list: List[str], phase: str) -> List[Tuple[Path, int]]:
        rows: List[Tuple[Path,int]] = []
        for key in id_list:
            item = identities[key]
            rows.append((item["real"], 0))
            k = 1 if (only_rvfa_flag and cross_manip is None) else fake_mult
            fks = choose_fakes_for_identity(
                item=item,
                k=k,
                seed=args.seed,
                key=key,
                only_rvfa=(only_rvfa_flag and cross_manip is None),
                phase=phase,
                cross_manip=cross_manip,
            )
            for fp in fks:
                rows.append((fp, 1))
        return rows

    train_rows = build_rows(train_ids, phase="train")
    test_rows  = build_rows(test_ids,  phase="test")

    # 안전장치: 비어있으면 에러
    if not train_rows or not test_rows:
        raise RuntimeError(f"Empty output: train_rows={len(train_rows)}, test_rows={len(test_rows)}")

    write_csv(out_dir / "trainset.csv", train_rows)
    write_csv(out_dir / "testset.csv",  test_rows)

    def count_stats(rows):
        r = sum(1 for _,lab in rows if lab==0)
        f = sum(1 for _,lab in rows if lab==1)
        return r, f

    tr_r, tr_f = count_stats(train_rows)
    te_r, te_f = count_stats(test_rows)

    print("=== Done ===")
    print(f"Train: real={tr_r}, fake={tr_f}, ratio fake/real={tr_f/max(1,tr_r):.2f}")
    print(f"Test : real={te_r}, fake={te_f}, ratio fake/real={te_f/max(1,te_r):.2f}")
    print(f"CSV saved to: {out_dir}")

if __name__ == "__main__":
    main()

