import argparse
import csv
import hashlib
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

# 유효 조작 카테고리 정의 (대소문자 무시용으로 upper() 사용)
VALID_CATS = ("RVFA", "FVRA", "FVFA")

# -------- 경로 치환 유틸 --------
OLD_DATA_ANCHOR = "/AVDFD/data/"
OLD_PREFIX_ROOT = "/data/jhlee39/workspace/repos"   # 안전한 앞부분 식별
NEW_BASE = "/data2/local_datasets/jhlee39"          # 여기에 이어 붙임

def rewrite_to_data2(p: Path) -> Path:
    """
    AVDFD/data 앞부분을 잘라내고 /data2/local_datasets/jhlee39/ 로 치환.
    이미 data2 경로이면 그대로 둠.
    """
    s = str(p)
    if s.startswith(NEW_BASE):
        return p  # 이미 목표 경로
    if OLD_DATA_ANCHOR in s:
        # 앞부분(…/AVDFD/data/)까지 자르고 뒤를 NEW_BASE에 붙인다.
        before, after = s.split(OLD_DATA_ANCHOR, maxsplit=1)
        # safety: before가 OLD_PREFIX_ROOT로 시작할 때만 치환
        if before.startswith(OLD_PREFIX_ROOT):
            return Path(NEW_BASE) / after
    # 혹시 과거의 하드코딩 치환이 남아있을 수도 있으니 보조 치환
    s2 = s.replace(
        "/data/jhlee39/workspace/repos/AVDFD/data/",
        NEW_BASE + "/"
    )
    return Path(s2)

# -------- 파일 탐색 유틸 --------
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

# -------- 인덱스 빌드 --------
def build_identity_index(data_root: Path, vc_mode: str, only_rvfa_build: bool) -> Dict[str, dict]:
    """
    - 항상 RVRA(RealVideo-RealAudio)는 real로 사용
    - only_rvfa_build=True면 fake는 RVFA 계열(VC_MODE에 따라 경로 상이)만 인덱싱
    - 아니면 RVFA/FVRA/FVFA 모두 인덱싱
    """
    RVRA = data_root / "RealVideo-RealAudio"

    # RVFA 경로 선택 (VC 난이도)
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
        if not race_dir.is_dir(): 
            continue
        for gender_dir in sorted(race_dir.iterdir()):
            if not gender_dir.is_dir():
                continue
            for id_dir in sorted(gender_dir.iterdir()):
                if not id_dir.is_dir():
                    continue
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
            if not race_dir.is_dir(): 
                continue
            for gender_dir in sorted(race_dir.iterdir()):
                if not gender_dir.is_dir(): 
                    continue
                for id_dir in sorted(gender_dir.iterdir()):
                    if not id_dir.is_dir():
                        continue
                    key = f"{race_dir.name}/{gender_dir.name}/{id_dir.name}"
                    if key not in identities:
                        continue
                    files = list_all_mp4s(id_dir)
                    if files:
                        identities[key]["fakes"][bucket_name].extend([p.resolve() for p in files])

    # only_rvfa_build이면 RVFA만, 아니면 전체 조작을 인덱스
    attach_fakes(RVFA, "RVFA")
    if not only_rvfa_build:
        attach_fakes(FVRA, "FVRA")
        attach_fakes(FVFA, "FVFA")

    # 최소 1개의 fake가 존재하는 identity만 남김
    filtered = {k: v for k, v in identities.items()
                if sum(len(v["fakes"][cat]) for cat in VALID_CATS) > 0}
    return filtered

# -------- split --------
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

# -------- 샘플 선택 --------
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

    # 교차-조작(leave-one-manip-out) 설정
    if cross_manip:
        # test는 지정 조작만, train은 지정 조작을 제외
        active_cats = [cross_manip] if phase == "test" else [c for c in VALID_CATS if c != cross_manip]
    else:
        active_cats = ["RVFA"] if only_rvfa else list(VALID_CATS)

    # 실제로 샘플이 존재하는 조작만 유지
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
    while len(out) < k and order:
        pool = pools[order[0]]
        if not pool:
            break
        out.append(pool[len(out) % len(pool)])
    return out

# -------- CSV 저장 --------
def write_csv(save_path: Path, rows: List[Tuple[Path, int]]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "target"])
        for p, lab in rows:
            mapped = rewrite_to_data2(p.resolve())
            writer.writerow([str(mapped), lab])

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--dir-name", type=str, default="", help="OUT_DIR/<dir-name>/ 아래에 CSV 저장 (없으면 OUT_DIR 에 바로 저장)")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--fake-mult", type=int, default=3)
    ap.add_argument("--only-RVFA", action="store_true")
    ap.add_argument("--vc", type=str, default="false", choices=["false", "harder", "hardest"])
    ap.add_argument("--cross-manip", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if args.dir_name:
        out_dir = out_dir / args.dir_name

    # only_RVFA는 cross_manip이 비어있을 때만 의미 있게 작동(스크립트에서도 분기)
    only_rvfa_build = args.only_RVFA and args.cross_manip.strip() == ""
    vc_mode = args.vc
    cross_manip = args.cross_manip.strip().upper() or None

    print(f"[INFO] Using vc_mode = {vc_mode}")
    print(f"[INFO] only_RVFA    = {only_rvfa_build}")
    print(f"[INFO] cross_manip  = {cross_manip or 'None'}")

    identities = build_identity_index(data_root, vc_mode=vc_mode, only_rvfa_build=only_rvfa_build)
    train_ids, test_ids = stratified_identity_split(identities, train_ratio=args.train_ratio, seed=args.seed)

    def build_rows(id_list, phase):
        rows = []
        for key in id_list:
            item = identities[key]
            # real 1개
            rows.append((item["real"], 0))
            # fake k개: only_rvfa면 k=1, 아니면 fake_mult
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
    print(f"Output dir: {out_dir}")
    print(f"Train: real={sum(1 for _,l in train_rows if l==0)}, fake={sum(1 for _,l in train_rows if l==1)}")
    print(f"Test : real={sum(1 for _,l in test_rows if l==0)}, fake={sum(1 for _,l in test_rows if l==1)}")

if __name__ == "__main__":
    main()
