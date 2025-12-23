# Voice-Cloned Deepfake Detection with Refined Cross-Modal Hard Samples

Recent advances in voice cloning enable the creation of highly synchronized yet semantically fake audio, which significantly weakens conventional audio-visual deepfake detectors relying on lip–speech misalignment.
This repository provides a **two-stage training framework** designed to explicitly target such **voice-cloned deepfakes** by focusing on **subtle cross-modal inconsistencies** rather than temporal asynchrony.

To support this goal, we construct **Refined RealVideo–FakeAudio (RVFA)** samples from FakeAVCeleb using a multi-inference voice-cloning pipeline, and decompose them into **audio and visual elements** for efficient and scalable training.

---

## Repository Overview

### Top-Level Structure

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

## Core Pipeline Overview

This project consists of **three main stages**:

1. **Data Element Extraction (Preprocessing)**
2. **Stage-1 Representation Learning (Pretraining)**
3. **Stage-2 Fine-Tuning / Detection Training**

Additionally, a **baseline pipeline (AVFF)** is provided for comparison.

---

## Step 0. Data Preprocessing (Element Extraction)

### Purpose

The `data/` directory contains preprocessing scripts that **convert refined fake videos** (generated via multi-inference voice cloning) into **separable audio-visual elements**.

Instead of using raw video files during training, we:
- extract **video frames**
- extract **mel-spectrogram frames**
- store them as reusable elements

All extracted elements are saved under `data/elements/` and reused for all subsequent training stages.

### Scripts

```
data/
├── preprocess.py
├── preprocess.sh
└── elements/
```

---

## Step 1. Baseline Execution (AVFF)

### Purpose

`baseline.py` runs the **original AVFF baseline** to provide a fair reference point.

- Based on **OpenAVFF (CVPR 2024)**
- No architectural modification
- Used only for comparison

### Scripts

```
baseline.py
baseline.sh
```

### Usage

```bash
bash baseline.sh
```

---

## Step 2. Stage-1 Training (Pretraining)

### Purpose

`pretrain.py` implements **Stage-1 representation learning**, where the model learns cross-modal embeddings that emphasize subtle inconsistencies introduced by voice cloning.

- Uses element-level inputs (mel frames + video frames)
- Focuses on representation quality
- Produces a reusable Stage-1 checkpoint

### Scripts

```
pretrain.py
pretrain.sh
```

### Usage

```bash
bash pretrain.sh
```

---

## Step 3. Stage-2 Training (Fine-Tuning / Detection)

### Purpose

`train.py` performs **Stage-2 detection training** using the pretrained Stage-1 model.

- Supports frozen or fine-tuned Stage-1
- Trains the final detection head

### Scripts

```
train.py
train.sh
```

### Usage

```bash
bash train.sh
```

---

## Configuration

All hyperparameters are controlled via JSON files in the `config/` directory.

```
config/
├── pretrain.json
├── train.json
└── ...
```

This allows clean experiment control without modifying code.

---

## Summary

- **data/**: element-level preprocessing
- **baseline.py**: AVFF baseline
- **pretrain.py**: Stage-1 representation learning
- **train.py**: Stage-2 detection training
- **config/**: hyperparameter management

This design cleanly separates **data processing**, **representation learning**, and **detection**, and is specifically tailored for **hard voice-cloned deepfake detection**.
ub 