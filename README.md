# Directionality-aware Audio-Visual Deepfake Detection Considering Cross-modal Asymmetry

## Abstract

Conventional audio-visual deepfake detection methods primarily rely on detecting synchronization inconsistencies between modalities, typically treating audio and visual streams as symmetric signals that should be mutually aligned or reconstructed in a one-to-one manner. However, in real speech, the relationship between audio and visual modalities is inherently asymmetric: while audio strongly constrains visual articulation, predicting audio solely from visual information is fundamentally ill-posed and unstable.

To address this limitation, we propose a **directionality-aware, one-way audio-visual deepfake detection framework** that verifies visual consistency conditioned on audio. Specifically, we first learn an audio-only representation that is robust to generative artifacts through unimodal representation learning. Then, conditioned on audio features, the model predicts time-aligned visual representations and measures their discrepancy from actual visual embeddings as a detection signal. Furthermore, we introduce a confidence-aware fusion strategy that adaptively weights audio-only cues and cross-modal consistency cues, allowing the model to emphasize the more reliable evidence on a per-sample basis.

Experimental results show that the audio-only branch already achieves strong detection performance, and that multi-modal fusion further improves robustness under challenging conditions where synchronization cues are weak. These results demonstrate that directionality-aware modeling is an effective principle for detecting high-quality, voice-cloned deepfakes.

---

## Repository Overview

```
.
├── baseline.py
├── baseline.sh
├── pretrain.py
├── pretrain.sh
├── train.py
├── train.sh
│
├── config/
│   └── *.json
│
├── data/
│   ├── preprocess.py
│   ├── preprocess.sh
│   └── elements/
│
└── OpenAVFF/
```

---

## Overall Pipeline

1. Dataset preparation (VoxCeleb2 + FakeAVCeleb)
2. Voice cloning and hard sample generation (Seed-VC)
3. Element-level preprocessing (audio / visual decomposition)
4. Baseline execution (AVFF)
5. Stage-1 audio representation learning
6. Stage-2 directionality-aware audio-visual detection

---

## Step 0. Environment Setup (Preprocessing)

```bash
cd data/preprocessing
conda create -n preprocessing python=3.10 -y
conda activate preprocessing
```

### Recommended PyTorch Version (CUDA 12.x)

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```

Do **not** install additional requirements at this stage.

---

## Step 1. Dataset Download

### 1-1. VoxCeleb2 (Selective Download)

```bash
bash down_vox2.sh
```

Optional full download:

```bash
bash voxceleb2_download.sh
conda install -c conda-forge hf_transfer -y
```

---

### 1-2. FakeAVCeleb

Download **FakeAVCeleb_v1.2** into:

```
data/FakeAVCeleb_Refine/
```

---

## Step 2. Voice Cloning (Seed-VC)

```bash
cd data/preprocessing
git clone https://github.com/Plachtaa/seed-vc.git
pip install -r seed-vc/requirements.txt
pip install datasets>=2.20.0 tqdm>=4.66.0 speechbrain
```

Move inference scripts:

```bash
mv data/preprocessing/multi_inference_v2.py seed-vc/
mv data/preprocessing/multi_inference_v2.sh seed-vc/
```

Run inference:

```bash
cd seed-vc
sbatch multi_inference_v2.sh
```

---

## Step 3. Element-level Preprocessing

```bash
cd data
bash preprocess.sh
```

Outputs are stored in `data/elements/`.

---

## Step 4. Main Environment Setup (AVDFD)

```bash
conda create -n AVDFD python=3.10 -y
conda activate AVDFD
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r OpenAVFF/requirements.txt
```

---

## Step 5. Baseline Execution

```bash
bash baseline.sh
```

---

## Step 6. Stage-1 Training

```bash
bash pretrain.sh
```

---

## Step 7. Stage-2 Training

```bash
bash train.sh
```

---

## Configuration

All hyperparameters are defined in JSON files under `config/`.

---

## Summary

This repository implements a directionality-aware audio-visual deepfake detection framework that explicitly accounts for cross-modal asymmetry and is robust to high-quality voice-cloned deepfakes.
