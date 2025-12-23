# pretrain.py
import argparse, math, os, random, tempfile
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from src.dataset import (
    read_identities,
    build_fake_relkeys_for_pretrain,
    FakeAVCelebPretrainPairs,
)
from src.model import AudioPretrainNet


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


def supcon_loss(z, labels, temperature=0.07):
    """
    Supervised contrastive loss.
    z: [B,D] normalized
    labels: [B]
    """
    B = z.size(0)
    sim = torch.matmul(z, z.t()) / temperature
    sim = sim - torch.max(sim, dim=1, keepdim=True)[0]

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(z.device)
    logits_mask = 1 - torch.eye(B, device=z.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    return -mean_log_prob_pos.mean()


def ordinal_ring_loss(z, labels, proto_real, margin=0.2):
    """
    Real prototype(proto_real)를 기준으로,
    RVRA(0) < RVFA-SVC(1) < RVFA-VC(2) < RVFA(3)
    순서로 거리의 평균이 margin 이상씩 증가하도록 강제하는 loss.

    z: [B, D] (contrastive embedding)
    labels: [B] (0..3)
    proto_real: [D]
    """
    if z.numel() == 0:
        return z.new_tensor(0.0)

    proto = proto_real.to(z.device)
    d = torch.norm(z - proto.unsqueeze(0), dim=1)  # [B]

    losses = []
    # labels: 0,1,2,3 (ordinal)
    max_level = int(labels.max().item()) if labels.numel() > 0 else 0
    max_level = min(max_level, 3)

    for k in range(max_level):
        mask_k = (labels == k)
        mask_k1 = (labels == k + 1)
        if mask_k.any() and mask_k1.any():
            d_k = d[mask_k].mean()
            d_k1 = d[mask_k1].mean()
            # we want d_k1 - d_k >= margin  →  loss = ReLU(margin - (d_k1 - d_k))
            diff = d_k1 - d_k
            loss_k = torch.relu(margin - diff)
            losses.append(loss_k)

    if len(losses) == 0:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_one_epoch(model, loader, optim, device, scaler, use_amp=True, grad_clip=1.0,
                    lambda_supcon=1.0, lambda_aux=1.0,
                    lambda_ord=1.0, margin_ord=0.2):
    model.train()
    bce = nn.BCEWithLogitsLoss()

    total, total_loss = 0, 0.0
    labs_bin, probs_bin = [], []

    pbar = tqdm(loader, desc="Pretrain", ncols=100, leave=False)
    for mel_real, mel_fake, cls_lab_fake, bin_lab_fake, _ in pbar:
        optim.zero_grad(set_to_none=True)

        mel_real = mel_real.to(device)
        mel_fake = mel_fake.to(device)
        cls_lab_fake = cls_lab_fake.to(device)
        bin_lab_fake = bin_lab_fake.to(device)

        # [B,128,768] 가정. 혹시 [128,768]이면 batch 차원 추가.
        mel_real = mel_real.unsqueeze(1) if mel_real.ndim == 2 else mel_real
        mel_fake = mel_fake.unsqueeze(1) if mel_fake.ndim == 2 else mel_fake

        mel = torch.cat([mel_real, mel_fake], dim=0)

        # ordinal label: RVRA(0) + fake classes(1..3)
        real_lab = torch.zeros_like(cls_lab_fake)
        cls_lab = torch.cat([real_lab, cls_lab_fake], dim=0)

        # binary label: real=0, fake=1
        bin_lab = torch.cat([torch.zeros_like(bin_lab_fake), bin_lab_fake], dim=0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            _, z, logit_aux = model(mel)
            loss_con = supcon_loss(z, cls_lab)
            loss_aux = bce(logit_aux, bin_lab)
            loss_ord = ordinal_ring_loss(z, cls_lab, model.proto_real, margin=margin_ord)
            loss = (lambda_supcon * loss_con +
                    lambda_aux * loss_aux +
                    lambda_ord * loss_ord)

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
            prob_aux = torch.sigmoid(logit_aux)
            labs_bin.extend(bin_lab.detach().cpu().tolist())
            probs_bin.extend(prob_aux.detach().cpu().tolist())

        total_loss += loss.item() * mel.size(0)
        total += mel.size(0)
        pbar.set_postfix(
            loss=total_loss / total,
            con=loss_con.item(),
            aux=loss_aux.item(),
            ord=loss_ord.item(),
        )

    auc, ap = compute_auc_ap(labs_bin, probs_bin)
    return total_loss / total, auc, ap


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True,
             lambda_supcon=1.0, lambda_aux=1.0,
             lambda_ord=1.0, margin_ord=0.2):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total, total_loss = 0, 0.0
    labs_bin, probs_bin = [], []

    pbar = tqdm(loader, desc="Val", ncols=100, leave=False)
    for mel_real, mel_fake, cls_lab_fake, bin_lab_fake, _ in pbar:
        mel_real = mel_real.to(device)
        mel_fake = mel_fake.to(device)
        cls_lab_fake = cls_lab_fake.to(device)
        bin_lab_fake = bin_lab_fake.to(device)

        mel_real = mel_real.unsqueeze(1) if mel_real.ndim == 2 else mel_real
        mel_fake = mel_fake.unsqueeze(1) if mel_fake.ndim == 2 else mel_fake
        mel = torch.cat([mel_real, mel_fake], dim=0)

        real_lab = torch.zeros_like(cls_lab_fake)
        cls_lab = torch.cat([real_lab, cls_lab_fake], dim=0)
        bin_lab = torch.cat([torch.zeros_like(bin_lab_fake), bin_lab_fake], dim=0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            _, z, logit_aux = model(mel)
            loss_con = supcon_loss(z, cls_lab)
            loss_aux = bce(logit_aux, bin_lab)
            loss_ord = ordinal_ring_loss(z, cls_lab, model.proto_real, margin=margin_ord)
            loss = (lambda_supcon * loss_con +
                    lambda_aux * loss_aux +
                    lambda_ord * loss_ord)

        prob_aux = torch.sigmoid(logit_aux)
        labs_bin.extend(bin_lab.detach().cpu().tolist())
        probs_bin.extend(prob_aux.detach().cpu().tolist())

        total_loss += loss.item() * mel.size(0)
        total += mel.size(0)
        pbar.set_postfix(loss=total_loss / total)

    auc, ap = compute_auc_ap(labs_bin, probs_bin)
    return total_loss / total, auc, ap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_root", type=str, required=True)
    p.add_argument("--split_dir", type=str, default="data/splits")
    p.add_argument("--train_split", type=str, required=True)
    p.add_argument("--val_split", type=str, required=True)
    p.add_argument("--train_fakes", type=str, default="RVFA-VC,RVFA-SVC,RVFA")
    p.add_argument("--val_fakes", type=str, default="RVFA-VC,RVFA-SVC,RVFA")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr_ratio", type=float, default=1 / 50.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--lambda_supcon", type=float, default=1.0)
    p.add_argument("--lambda_aux", type=float, default=1.0)
    p.add_argument("--lambda_ord", type=float, default=1.0)
    p.add_argument("--margin_ord", type=float, default=0.2)

    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save_path", type=str, default="audio_pretrained.pt")
    p.add_argument(
        "--no_save",
        action="store_true",
        help="If set, do not save any model checkpoint to disk.",
    )

    # DIRE mode: True -> use mel_err, False -> use mel
    p.add_argument(
        "--dire_mode",
        action="store_true",
        help="If set, load from 'mel_err'; otherwise load from 'mel'.",
    )

    args = p.parse_args()

    tempfile.tempdir = os.environ.get("TMPDIR", "/tmp")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    # audio subdir 선택
    audio_subdir = "mel_err" if args.dire_mode else "mel"
    print(f"[Pretrain] audio_subdir = {audio_subdir}")

    split_dir = Path(args.split_dir)
    train_ids = read_identities(str(split_dir / args.train_split))
    val_ids = read_identities(str(split_dir / args.val_split))

    train_fakes = [s.strip() for s in args.train_fakes.split(",") if s.strip()]
    val_fakes = [s.strip() for s in args.val_fakes.split(",") if s.strip()]

    train_fake_relkeys = build_fake_relkeys_for_pretrain(
        args.preproc_root, train_ids, train_fakes, audio_subdir=audio_subdir
    )
    val_fake_relkeys = build_fake_relkeys_for_pretrain(
        args.preproc_root, val_ids, val_fakes, audio_subdir=audio_subdir
    )

    train_ds = FakeAVCelebPretrainPairs(
        args.preproc_root, train_fake_relkeys, seed=args.seed, audio_subdir=audio_subdir
    )
    val_ds = FakeAVCelebPretrainPairs(
        args.preproc_root, val_fake_relkeys, seed=args.seed, audio_subdir=audio_subdir
    )

    tr_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=1 if args.num_workers > 0 else None,
    )
    va_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=1 if args.num_workers > 0 else None,
    )

    model = AudioPretrainNet().to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    min_lr = args.lr * args.min_lr_ratio
    sched = build_warmup_cosine_scheduler(
        optim, args.epochs, args.warmup_epochs, args.lr, min_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_ap = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_auc, tr_ap = train_one_epoch(
            model,
            tr_loader,
            optim,
            device,
            scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            lambda_supcon=args.lambda_supcon,
            lambda_aux=args.lambda_aux,
            lambda_ord=args.lambda_ord,
            margin_ord=args.margin_ord,
        )
        va_loss, va_auc, va_ap = evaluate(
            model,
            va_loader,
            device,
            use_amp=use_amp,
            lambda_supcon=args.lambda_supcon,
            lambda_aux=args.lambda_aux,
            lambda_ord=args.lambda_ord,
            margin_ord=args.margin_ord,
        )
        sched.step()

        print(
            f"[Ep {ep:03d}] "
            f"train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} train_ap={tr_ap:.4f} | "
            f"val_loss={va_loss:.4f} val_auc={va_auc:.4f} val_ap={va_ap:.4f} | "
            f"lr={optim.param_groups[0]['lr']:.2e}"
        )

        if va_ap > best_ap:
            best_ap = va_ap
            best_state = copy.deepcopy(model.audio_enc.state_dict())
            print(f"  -> updated best audio_enc (val_ap={best_ap:.4f})")

    if (not args.no_save) and (best_state is not None):
        torch.save(best_state, args.save_path)
        print(f"[Done] Saved best audio_enc to {args.save_path} (val_ap={best_ap:.4f})")
    else:
        if args.no_save:
            print("[Done] Training finished with no_save=True (no checkpoint written).")
        else:
            print("[Done] Training finished but no valid best_state was recorded.")


if __name__ == "__main__":
    main()
