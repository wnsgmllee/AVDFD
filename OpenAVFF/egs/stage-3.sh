#!/usr/bin/bash 
#SBATCH -J AVFF
#SBATCH --gres=gpu:2      
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 5-23
#SBATCH -o logs/slurm-%A.out
#SBATCH --nodelist=ariel-v6

# multiprocessing을 위한 temp directory 설정 (Local storage로)
JOB_LOCAL_TMP=/dev/shm/$SLURM_JOB_ID
mkdir -p "$JOB_LOCAL_TMP"
export TMPDIR="$JOB_LOCAL_TMP"
export TEMP="$JOB_LOCAL_TMP"
export TMP="$JOB_LOCAL_TMP"

contrast_loss_weight=0.01
mae_loss_weight=1.0
norm_pix_loss=True

# you can use any checkpoints with a decoder, but by default, we use vision-MAE checkpoint
pretrain_path=../checkpoints/stage3_init_from_stage2.pth

lr=1e-5
head_lr=20
epoch=10
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa_start=1
wa_end=10
target_length=1024
noise=True
batch_size=4
lr_adapt=False

n_print_steps=100


# rvfa_hardest_0.8 all_base_0.8 all_harder_0.8 CM_RVFA_harder_0.8 rvfa_hardest_0.8
tr_data=../data/all_base_0.8/trainset.csv 
te_data=../data/all_base_0.8/testset.csv 

# exp_dir=./exp/self-pretrain
save_dir=../checkpoints/
mkdir -p $save_dir
mkdir -p ${save_dir}/models

CUDA_CACHE_DISABLE=1 python -W ignore ../src/run_ft.py \
--data-train ${tr_data} --data-val ${te_data} --save-dir $save_dir --n_classes 2 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--target_length ${target_length} --noise ${noise} \
--lr_adapt ${lr_adapt} \
--norm_pix_loss ${norm_pix_loss} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--loss BCE --metrics mAP --warmup True \
--wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--head_lr ${head_lr} \
--pretrain_path ${pretrain_path} --num_workers 6\