# train.py
import argparse, math, os, random, tempfile, json, itertools
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from src.dataset import read_identities, build_key_list_dual, FakeAVCelebDual
from src.model import AVDetector

from marlin_pytorch import Marlin


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_auc_ap(labels, probs):
    y = np.array(labels, np.float32)
    p = np.array(probs, np.float32)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, p), average_precision_score(y, p)


def build_warmup_cosine_scheduler(optim, epochs, warmup_epochs, base_lr, min_lr):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        t = (ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return min_lr / base_lr + 0.5 * (1.0 - min_lr / base_lr) * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


# ---------------------------
# DDP init
# ---------------------------
def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        if rank == 0:
            print(f"[DDP] world_size={get_world_size()}")
        return device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def all_gather_list(local_list):
    if get_world_size() == 1:
        return local_list
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local_list)
    out = []
    for g in gathered:
        out.extend(g)
    return out


# ---------------------------
# Align / Repel loss (핵심 추가)
# ---------------------------
def a2v_align_repel_loss(mismatch_mse: torch.Tensor, y_video: torch.Tensor, margin: float):
    """
    mismatch_mse: [B]  (pred_v_seq vs v_seq MSE)
    y_video:      [B]  (0 real, 1 fake)  - 현재 파이프라인에서는 fakes는 전부 1로 들어옴

    real(0):  minimize mse
    fake(1):  encourage mse >= margin  (hinge)
    """
    y = y_video.view(-1).float()
    real_mask = (y < 0.5).float()  # [B]
    fake_mask = (y >= 0.5).float()

    loss_real = (mismatch_mse * real_mask)  # mse
    loss_fake = (torch.relu(margin - mismatch_mse) * fake_mask)  # hinge

    denom = (real_mask.sum() + fake_mask.sum()).clamp_min(1.0)
    return (loss_real.sum() + loss_fake.sum()) / denom


# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(
    model,
    loader,
    optim,
    device,
    scaler,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    prefix: str = "",
    lambda_align: float = 0.0,
    margin_align: float = 0.5,
):
    """
    (수정)
    - 기존: BCE(logit_fusion, y_video)만 사용
    - 추가: A2V 복원/불일치 제어 loss를 함께 사용
        L = BCE + lambda_align * AlignRepel(mismatch_mse, y_video)
    """
    model.train()
    bce = nn.BCEWithLogitsLoss()

    sum_loss = torch.zeros(1, device=device)
    sum_count = torch.zeros(1, device=device)

    local_labs, local_probs = [], []

    desc = f"{prefix}Train" if prefix else "Train"
    pbar = tqdm(loader, desc=desc, ncols=110, leave=False, disable=(get_rank() != 0))

    for mel, v_clip, first_frame, y_video, _y_audio, _, _ in pbar:
        optim.zero_grad(set_to_none=True)

        mel = mel.to(device)
        v_clip = v_clip.to(device)
        first_frame = first_frame.to(device)
        y_video = y_video.to(device)

        if mel.ndim == 2:
            mel = mel.unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            _logit_audio, logit_fusion, aux = model(mel, v_clip, first_frame, return_aux=True)

            loss_cls = bce(logit_fusion, y_video)

            if lambda_align > 0:
                mismatch_mse = aux["mismatch_mse"]  # [B]
                loss_align = a2v_align_repel_loss(mismatch_mse, y_video, margin=margin_align)
                loss = loss_cls + lambda_align * loss_align
            else:
                loss_align = None
                loss = loss_cls

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        with torch.no_grad():
            prob = torch.sigmoid(logit_fusion)
            local_labs.extend(y_video.detach().cpu().tolist())
            local_probs.extend(prob.detach().cpu().tolist())

        bs = y_video.size(0)
        sum_loss += loss.detach() * bs
        sum_count += bs

        if get_rank() == 0:
            if loss_align is None:
                pbar.set_postfix(loss=(sum_loss / sum_count).item(), cls=loss_cls.item())
            else:
                pbar.set_postfix(loss=(sum_loss / sum_count).item(), cls=loss_cls.item(), align=loss_align.item())

    if is_dist():
        dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_count, op=dist.ReduceOp.SUM)
    loss_global = (sum_loss / sum_count).item()

    labs = all_gather_list(local_labs)
    probs = all_gather_list(local_probs)

    auc, ap = compute_auc_ap(labs, probs)
    return loss_global, auc, ap


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool = True, prefix: str = "",
             lambda_align: float = 0.0, margin_align: float = 0.5):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    sum_loss = torch.zeros(1, device=device)
    sum_count = torch.zeros(1, device=device)

    local_labs, local_probs = [], []

    desc = f"{prefix}Val" if prefix else "Val"
    pbar = tqdm(loader, desc=desc, ncols=110, leave=False, disable=(get_rank() != 0))

    for mel, v_clip, first_frame, y_video, _y_audio, _, _ in pbar:
        mel = mel.to(device)
        v_clip = v_clip.to(device)
        first_frame = first_frame.to(device)
        y_video = y_video.to(device)

        if mel.ndim == 2:
            mel = mel.unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            _logit_audio, logit_fusion, aux = model(mel, v_clip, first_frame, return_aux=True)
            loss_cls = bce(logit_fusion, y_video)

            if lambda_align > 0:
                mismatch_mse = aux["mismatch_mse"]
                loss_align = a2v_align_repel_loss(mismatch_mse, y_video, margin=margin_align)
                loss = loss_cls + lambda_align * loss_align
            else:
                loss = loss_cls

        prob = torch.sigmoid(logit_fusion)
        local_labs.extend(y_video.detach().cpu().tolist())
        local_probs.extend(prob.detach().cpu().tolist())

        bs = y_video.size(0)
        sum_loss += loss.detach() * bs
        sum_count += bs

        if get_rank() == 0:
            pbar.set_postfix(loss=(sum_loss / sum_count).item())

    if is_dist():
        dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_count, op=dist.ReduceOp.SUM)
    loss_global = (sum_loss / sum_count).item()

    labs = all_gather_list(local_labs)
    probs = all_gather_list(local_probs)

    auc, ap = compute_auc_ap(labs, probs)
    return loss_global, auc, ap


# ---------------------------
# Single experiment runner
# ---------------------------
def build_model_and_optim(args, device):
    model = AVDetector(
        marlin_name=args.marlin_name,
        freeze_vis_backbone=args.freeze_vis_enc,
        a2v_nhead=args.a2v_nhead,
        a2v_layers=args.a2v_layers,
        d_model=512,
    ).to(device)

    if args.audio_pretrained and Path(args.audio_pretrained).exists():
        sd = torch.load(args.audio_pretrained, map_location="cpu")
        missing, unexpected = model.audio_enc.load_state_dict(sd, strict=False)
        if get_rank() == 0:
            print(f"[Load] audio_enc from {args.audio_pretrained}")
            if missing:
                print(f"  missing keys: {missing}")
            if unexpected:
                print(f"  unexpected keys: {unexpected}")

    ddp_wrapped = False
    if get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index
        )
        ddp_wrapped = True

    raw_model = model.module if ddp_wrapped else model

    backbone_params = []
    new_params = []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if "vis_enc.backbone" in name:
            backbone_params.append(p)
        else:
            new_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr * args.backbone_lr_scale})
    if new_params:
        param_groups.append({"params": new_params, "lr": args.lr})

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found for optimizer.")

    optim = torch.optim.AdamW(param_groups, weight_decay=0.01)

    min_lr = args.lr * args.min_lr_ratio
    sched = build_warmup_cosine_scheduler(optim, args.epochs, args.warmup_epochs, args.lr, min_lr)

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    return model, optim, sched, scaler, use_amp


def run_training(args, device, train_ds, val_ds, exp_name: str | None = None):
    prefix = f"[{exp_name}] " if exp_name else ""

    if get_world_size() > 1:
        tr_sampler = DistributedSampler(train_ds, shuffle=True)
        va_sampler = DistributedSampler(val_ds, shuffle=False)
        shuffle_tr = False
        shuffle_va = False
    else:
        tr_sampler = None
        va_sampler = None
        shuffle_tr = True
        shuffle_va = False

    tr_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle_tr, sampler=tr_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=1 if args.num_workers > 0 else None,
    )
    va_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=shuffle_va, sampler=va_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=1 if args.num_workers > 0 else None,
    )

    model, optim, sched, scaler, use_amp = build_model_and_optim(args, device)

    best_ap = -1.0
    best_auc = float("nan")
    bad = 0

    for ep in range(1, args.epochs + 1):
        if isinstance(tr_sampler, DistributedSampler):
            tr_sampler.set_epoch(ep)

        tr_loss, tr_auc, tr_ap = train_one_epoch(
            model, tr_loader, optim, device, scaler,
            use_amp=use_amp, grad_clip=args.grad_clip,
            prefix=prefix,
            lambda_align=args.lambda_align,
            margin_align=args.margin_align,
        )
        va_loss, va_auc, va_ap = evaluate(
            model, va_loader, device, use_amp=use_amp, prefix=prefix,
            lambda_align=args.lambda_align,
            margin_align=args.margin_align,
        )
        sched.step()

        if get_rank() == 0:
            print(
                f"{prefix}[Ep {ep:03d}] "
                f"train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} train_ap={tr_ap:.4f} | "
                f"val_loss={va_loss:.4f} val_auc={va_auc:.4f} val_ap={va_ap:.4f} | "
                f"lr_main={optim.param_groups[-1]['lr']:.2e}"
            )

        if va_ap > best_ap:
            best_ap = va_ap
            best_auc = va_auc
            bad = 0
            if get_rank() == 0:
                print(f"{prefix}  -> best val_ap updated: {best_ap:.4f}")
        else:
            bad += 1
            if get_rank() == 0:
                print(f"{prefix}  -> early stop counter {bad}/{args.patience}")
            if bad >= args.patience:
                if get_rank() == 0:
                    print(prefix + "[EarlyStop] triggered")
                break

    return best_auc, best_ap


# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_root", type=str, required=True)
    p.add_argument("--split_dir", type=str, default="data/splits")
    p.add_argument("--train_split", type=str, required=True)
    p.add_argument("--val_split", type=str, required=True)
    p.add_argument("--train_fakes", type=str, default="RVFA,FVRA,FVFA")
    p.add_argument("--val_fakes", type=str, default="RVFA-VC")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr_ratio", type=float, default=1 / 50.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--audio_pretrained", type=str, default="audio_pretrained.pt")
    p.add_argument("--patience", type=int, default=20)

    # (추가) A2V 복원/불일치 제어 하이퍼
    p.add_argument("--lambda_align", type=float, default=0.5,
                   help="Weight for A2V align/repel loss. 0 disables it.")
    p.add_argument("--margin_align", type=float, default=0.5,
                   help="For fake samples, encourage mismatch_mse >= margin_align.")

    # MARLIN visual encoder 옵션
    p.add_argument(
        "--marlin_name",
        type=str,
        default="marlin_vit_base_ytf",
        help="HuggingFace model name for MARLIN visual encoder.",
    )
    p.add_argument(
        "--freeze_vis_enc",
        action="store_true",
        help="If set, freeze MARLIN backbone parameters.",
    )

    # Backbone lr scale for finetuning
    p.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.1,
        help="LR multiplier for vis_enc.backbone (e.g., 0.1 means backbone lr = 0.1 * lr).",
    )

    # A2V 구조 하이퍼
    p.add_argument("--a2v_nhead", type=int, default=4)
    p.add_argument("--a2v_layers", type=int, default=2)

    # Hyperparameter tuning config (grid search)
    p.add_argument(
        "--tune_config",
        type=str,
        default=None,
        help="If set, JSON dict mapping param name -> list of candidate values (grid search).",
    )

    args = p.parse_args()

    tempfile.tempdir = os.environ.get("TMPDIR", "/tmp")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = ddp_init()

    # Rank 0에서만 MARLIN 캐시 체크/다운로드
    if get_rank() == 0:
        try:
            print(f"[Rank0] Trying Marlin.from_online('{args.marlin_name}') (first try)...")
            _ = Marlin.from_online(args.marlin_name)
            print("[Rank0] MARLIN loaded from cache/online (first try).")
        except Exception as e:
            print(f"[Rank0] First Marlin.from_online() failed: {e}")
            print("[Rank0] Cleaning MARLIN cache and retrying...")
            try:
                Marlin.clean_cache()
                _ = Marlin.from_online(args.marlin_name)
                print("[Rank0] MARLIN downloaded successfully after cleaning cache.")
            except Exception as e2:
                print(f"[Rank0] Second Marlin.from_online() failed: {e2}")
                raise RuntimeError(
                    "MARLIN weights could not be loaded even after cleaning cache. "
                    "인터넷/접속 권한 또는 GitHub 접근 문제를 확인해야 합니다."
                ) from e2

    if is_dist():
        dist.barrier()

    # Dataset 준비 (공통)
    split_dir = Path(args.split_dir)
    train_ids = read_identities(str(split_dir / args.train_split))
    val_ids = read_identities(str(split_dir / args.val_split))
    train_fakes = [s.strip() for s in args.train_fakes.split(",") if s.strip()]
    val_fakes = [s.strip() for s in args.val_fakes.split(",") if s.strip()]

    train_items = build_key_list_dual(args.preproc_root, train_ids, "RVRA", train_fakes)
    val_items = build_key_list_dual(args.preproc_root, val_ids, "RVRA", val_fakes)

    train_ds = FakeAVCelebDual(args.preproc_root, train_items, seed=args.seed)
    val_ds = FakeAVCelebDual(args.preproc_root, val_items, seed=args.seed)

    results = []

    if args.tune_config is None:
        best_auc, best_ap = run_training(args, device, train_ds, val_ds, exp_name=None)
        if get_rank() == 0:
            print(f"[Result] best_val_auc={best_auc:.4f}, best_val_ap={best_ap:.4f}")
    else:
        if get_rank() == 0:
            print(f"[TUNE] Loading hyper config (grid search) from: {args.tune_config}")
        cfg_path = Path(args.tune_config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"tune_config file not found: {args.tune_config}")

        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)

        if not isinstance(cfg_dict, dict):
            raise ValueError("tune_config JSON must be a dict: {param_name: [candidates,...], ...}")

        keys = sorted(cfg_dict.keys())
        value_lists = []
        for k in keys:
            vals = cfg_dict[k]
            if not isinstance(vals, list):
                vals = [vals]
            value_lists.append(vals)

        combos = list(itertools.product(*value_lists))

        if get_rank() == 0:
            print(f"[TUNE] Total combinations: {len(combos)}")
            print(f"[TUNE] Params: {keys}")

        for idx, combo in enumerate(combos):
            exp_args = deepcopy(args)
            exp_cfg = {}
            for k, v in zip(keys, combo):
                if hasattr(exp_args, k):
                    setattr(exp_args, k, v)
                    exp_cfg[k] = v
                else:
                    if get_rank() == 0:
                        print(f"[TUNE][WARN] args has no attribute '{k}', ignored in this combo.")

            exp_name = f"grid_{idx}"
            if get_rank() == 0:
                param_str = ", ".join(f"{k}: {exp_cfg.get(k)}" for k in keys)
                print(f"\n[TUNE] ===== Combo {idx} =====")
                print(f"[TUNE] {param_str}")

            best_auc, best_ap = run_training(exp_args, device, train_ds, val_ds, exp_name=exp_name)

            if get_rank() == 0:
                param_str = ", ".join(f"{k}: {exp_cfg.get(k)}" for k in keys)
                print(f"[TUNE] Combo {idx} result -> {param_str}, "
                      f"best_val_auc: {best_auc:.4f}, best_val_ap: {best_ap:.4f}")

            results.append(
                {
                    "index": idx,
                    "config": exp_cfg,
                    "best_val_auc": float(best_auc),
                    "best_val_ap": float(best_ap),
                }
            )

        if get_rank() == 0:
            out_path = cfg_path.with_name(cfg_path.stem + "_results.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "tune_config": str(cfg_path),
                        "params": keys,
                        "results": results,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"[TUNE] All grid-search results saved to: {out_path}")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
