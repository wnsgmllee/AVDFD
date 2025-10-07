#!/usr/bin/bash

#SBATCH -J Vox2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 0-23
#SBATCH -o logs/slurm-%A.out
#SBATCH --nodelist=ariel-v6


ID_ROOT="/data2/local_datasets/jhlee39/FakeAVCeleb_v1.2/RealVideo-RealAudio"
OUT_DIR="/data2/local_datasets/jhlee39/voxceleb_v2/data"



python down_vox2.py \
        --id-root "$ID_ROOT" \
	--out-dir "$OUT_DIR" 
