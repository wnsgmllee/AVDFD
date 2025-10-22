# run_ft.py (DDP-ready, NO MODEL SAVING)
import argparse
import os
import warnings
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloader import VideoAudioDataset
from models.video_cav_mae import VideoCAVMAEFT
from traintest_ft import train


def setup_distributed():
    """Return (world_size, local_rank, global_rank). Init if needed."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    global_rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return world_size, local_rank, global_rank


def is_rank0(world_size, global_rank):
    return (world_size == 1) or (global_rank == 0)


def safe_load_state_dict(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def main():
    parser = argparse.ArgumentParser(description='Video CAV-MAE (finetune)')
    parser.add_argument('--data-train', type=str, help='path to train data csv')
    parser.add_argument('--data-val', type=str, help='path to val data csv')
    parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
    parser.add_argument("--dataset_mean", default=-5.081, type=float, help="audio spec mean (for norm)")
    parser.add_argument("--dataset_std", default=4.4849, type=float, help="audio spec std (for norm)")
    parser.add_argument("--noise", default=False, type=bool, help="add noise to the input")

    parser.add_argument('--batch-size', default=32, type=int, help='batch size (per-process / per-GPU)')
    parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers (per-process)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate (backbone)')
    parser.add_argument("--lr_patience", type=int, default=1, help="epochs to wait before LR reduce if metric stalls")
    parser.add_argument("--metrics", type=str, default="mAP", choices=["mAP", "acc"])
    parser.add_argument("--loss", type=str, default="BCE", choices=["BCE", "CE"])
    parser.add_argument('--n-epochs', default=10, type=int)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--save-dir', default='checkpoints', type=str)
    parser.add_argument('--pretrain_path', default=None, type=str)
    parser.add_argument("--contrast_loss_weight", type=float, default=0.01)
    parser.add_argument("--mae_loss_weight", type=float, default=3.0)
    parser.add_argument('--save_model', action='store_true', default=False, help='(disabled by default)')
    parser.add_argument("--lrscheduler_start", default=10, type=int)
    parser.add_argument("--lrscheduler_step", default=5, type=int)
    parser.add_argument("--lrscheduler_decay", default=0.5, type=float)
    parser.add_argument("--lr_adapt", default=None)
    parser.add_argument('--norm_pix_loss', default=None)
    parser.add_argument("--n_print_steps", default=100, type=int)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--warmup', type=bool, default=True)
    # NOTE: 원 코드가 int로 되어 있었는데 비정상적으로 큼. float로 바꿔주는 게 맞음.
    parser.add_argument('--head_lr', type=float, default=5e-4)

    parser.add_argument("--wa_start", type=int, default=1)
    parser.add_argument("--wa_end", type=int, default=10)

    args = parser.parse_args()

    # ---- Absolutely disable model saving regardless of CLI ----
    # (1) Force flag off
    args.save_model = False

    # (2) Monkey-patch torch.save to prevent any checkpoint writes inside train()
    def _no_save(*_args, **_kwargs):
        pass  # swallow all save attempts silently
    torch.save = _no_save  # type: ignore

    # --- Distributed setup ---
    world_size, local_rank, global_rank = setup_distributed()
    rank0 = is_rank0(world_size, global_rank)

    # --- Repro/Perf niceties ---
    torch.backends.cudnn.benchmark = True

    im_res = 224
    audio_conf = {
        'num_mel_bins': 128, 'target_length': args.target_length,
        'freqm': args.freqm, 'timem': args.timem, 'mode': 'train',
        'mean': args.dataset_mean, 'std': args.dataset_std,
        'noise': args.noise, 'label_smooth': 0, 'im_res': im_res
    }
    val_audio_conf = {
        'num_mel_bins': 128, 'target_length': args.target_length,
        'freqm': 0, 'timem': 0, 'mixup': 0, 'mode': 'eval',
        'mean': args.dataset_mean, 'std': args.dataset_std,
        'noise': False, 'im_res': im_res
    }

    if rank0:
        print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(
            args.mae_loss_weight, args.contrast_loss_weight))
        print('[INFO] Model saving is DISABLED. No checkpoints will be written.')

    # --- Dataset & DataLoader ---
    train_ds = VideoAudioDataset(args.data_train, audio_conf, stage=2)
    val_ds   = VideoAudioDataset(args.data_val,   val_audio_conf, stage=2)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
        shuffle_flag_train = False
        shuffle_flag_val   = False
    else:
        train_sampler = None
        val_sampler   = None
        shuffle_flag_train = True
        shuffle_flag_val   = False

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle_flag_train,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=shuffle_flag_val,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=val_sampler
    )

    if rank0:
        # len(dataloader) = steps/epoch
        print(f"Using Train: {len(train_loader)}, Eval: {len(val_loader)}")

    # --- Model ---
    cavmae_ft = VideoCAVMAEFT(n_classes=args.n_classes)

    # --- Load pretrained (stage3-init-from-stage2) if provided ---
    if args.pretrain_path:
        missing, unexpected = safe_load_state_dict(cavmae_ft, args.pretrain_path)
        if rank0:
            print("Missing:", missing)
            print("Unexpected:", unexpected)
            print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(
                args.pretrain_path, len(missing), len(unexpected)))
    else:
        if rank0:
            warnings.warn("Note you are finetuning a model without any finetuning.")

    # --- Move to CUDA & wrap for DDP if multi-GPU ---
    cavmae_ft = cavmae_ft.cuda()
    if world_size > 1:
        cavmae_ft = torch.nn.parallel.DistributedDataParallel(
            cavmae_ft, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False
        )

    # --- Save dir (rank0 only) - kept for logs, not for checkpoints ---
    if rank0:
        print("\n Creating experiment directory (logs only): %s" % args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    # --- Train ---
    if rank0:
        print("Now start training for %d epochs" % args.n_epochs)
    train(cavmae_ft, train_loader, val_loader, args)

    # --- Cleanup ---
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
