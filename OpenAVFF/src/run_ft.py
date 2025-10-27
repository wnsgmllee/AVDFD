# run_ft.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataloader import VideoAudioDataset, VideoAudioEvalDataset, load_logmel_and_compute_stats
from models.video_cav_mae import VideoCAVMAEFT
from traintest_ft import train
import warnings

parser = argparse.ArgumentParser(description='Video CAV-MAE')
parser.add_argument('--data-train', type=str, help='path to train data csv')
parser.add_argument('--data-val', type=str, help='path to val data csv')

# Audio feature / normalization
parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
parser.add_argument("--dataset_mean", type=float, help="audio spec mean for normalization")
parser.add_argument("--dataset_std", type=float, help="audio spec std for normalization")
parser.add_argument("--noise", default=False, type=bool, help="add noise to the input")

# Loader / train
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument("--lr_patience", type=int, default=1, help="epochs to wait before lr reduce if no mAP improve")
parser.add_argument("--metrics", type=str, default="mAP", choices=["mAP", "acc"])
# >>> 변경: 기본 손실을 CE로 바꿔 모드 붕괴 완화
parser.add_argument("--loss", type=str, default="CE", choices=["BCE", "CE"])
parser.add_argument('--n-epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--n_classes', default=2, type=int, help='classes')

# (남겨두지만 이번 패치에서 저장 안함)
parser.add_argument('--save-dir', default='checkpoints', type=str, help='(unused in this patch) directory to save checkpoints')

# Pretrain / others
parser.add_argument('--pretrain_path', default=None, type=str, help='path to pretrain model')
parser.add_argument("--contrast_loss_weight", type=float, default=0.01)
parser.add_argument("--mae_loss_weight", type=float, default=3.0)
parser.add_argument("--lrscheduler_start", default=10, type=int)
parser.add_argument("--lrscheduler_step", default=5, type=int)
parser.add_argument("--lrscheduler_decay", default=0.5, type=float)
parser.add_argument("--lr_adapt")
parser.add_argument('--norm_pix_loss', default=None)
parser.add_argument("--n_print_steps", default=100, type=int)
parser.add_argument('--freqm', type=int, default=0)
parser.add_argument('--timem', type=int, default=0)
parser.add_argument('--warmup', type=bool, default=True)
# >>> 변경: head_lr 기본값 낮춤(초기 편향 폭주 완화)
parser.add_argument('--head_lr', type=int, default=10)
parser.add_argument("--wa_start", type=int, default=1)
parser.add_argument("--wa_end", type=int, default=10)

args = parser.parse_args()

# 1) Audio mean/std 자동 추정 (train csv 기반, 파이프라인과 동일 설정 사용)
print("↪ Automatically estimating dataset mean/std from training data...")
estimated_mean, estimated_std = load_logmel_and_compute_stats(args.data_train)
args.dataset_mean = estimated_mean
args.dataset_std = estimated_std
print(f"✔ Estimated mean: {estimated_mean:.4f}, std: {estimated_std:.4f}")

# 2) Audio configs
im_res = 224
audio_conf = {
    'num_mel_bins': 128,
    'target_length': args.target_length,
    'freqm': args.freqm, 'timem': args.timem,
    'mode': 'train',
    'mean': args.dataset_mean, 'std': args.dataset_std,
    'noise': args.noise,
    'label_smooth': 0,
    'im_res': im_res
}
val_audio_conf = {
    'num_mel_bins': 128,
    'target_length': args.target_length,
    'freqm': 0, 'timem': 0,
    'mixup': 0,
    'mode': 'eval',
    'mean': args.dataset_mean, 'std': args.dataset_std,
    'noise': False,
    'im_res': im_res
}

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))

# 3) Dataloaders
train_loader = DataLoader(
    VideoAudioDataset(args.data_train, audio_conf, stage=2),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True,
    drop_last=True, persistent_workers=True
)

# **오답 경로 기록을 위해 EvalDataset 사용 (경로 반환)**
val_loader = DataLoader(
    VideoAudioEvalDataset(args.data_val, val_audio_conf),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
    drop_last=False, persistent_workers=True
)

print(f"Using Train: {len(train_loader)*args.batch_size}, Eval: {len(val_loader)*args.batch_size}")

# 4) Model
cavmae_ft = VideoCAVMAEFT()

# (옵션) Pretrained init
if args.pretrain_path is not None:
    mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
    if not isinstance(cavmae_ft, torch.nn.DataParallel):
        cavmae_ft = torch.nn.DataParallel(cavmae_ft)
    miss, unexpected = cavmae_ft.load_state_dict(mdl_weight, strict=False)
    print("Missing: ", miss)
    print("Unexpected: ", unexpected)
    print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(
        args.pretrain_path, len(miss), len(unexpected)))
else:
    warnings.warn("Note you are finetuning a model without any pretrain weights.")

# 5) Train (모델 저장, result.csv 저장 모두 제거됨)
print("Now start training for %d epochs" % args.n_epochs)
train(cavmae_ft, train_loader, val_loader, args)
