# 🧩 Refined RVFA Dataset Preparation

This document describes how to generate the **Refined RealVideo-FakeAudio (RVFA)** dataset.  
The entire process is based on **Conda environments**, and after completing all steps,  
you will have a refined version of the RVFA dataset ready for training.

---

## 🧱 Step 0. Environment Setup

Move to the preprocessing directory and create the environment `AVDFD`.

```bash
cd data/preprocessing
conda create -n AVDFD python=3.10
conda activate AVDFD
pip install -r requirements.txt
```

> ⚙️ Recommended PyTorch version (for CUDA 12.x)
> ```bash
> pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
> ```

---

## 📦 Step 1. VoxCeleb2 Download (filtered by RVRA IDs)

Run the following script to download **only the speaker IDs** that exist in the  
`RealVideo-RealAudio (RVRA)` portion of FakeAVCeleb.  
This filtered download is less than **12 GB** in total.

```bash
bash down_vox2.sh
```

> ✅ **Selective download mode**  
> The script automatically scans RVRA to collect all unique IDs,  
> and downloads only the corresponding VoxCeleb2 utterances for those speakers.

---

### ⚠️ Downloading the full VoxCeleb2 dataset (optional)

If you want to store the **entire VoxCeleb2 dataset** locally  
(about **436 GB**, ~1 million utterances, 6 k speakers),  
you can run the following script:

```bash
bash voxceleb2_download.sh
```

> ⚠️ **Warning:** The full VoxCeleb2 dataset is extremely large.  
> It is **not recommended** unless you have sufficient SSD/NAS storage.  
> The selective 12 GB version is usually sufficient for all experiments.

---

## 🧬 Step 2. Voice Cloning (Seed-VC)

Refined RVFA is created using **Seed-VC (Voice Conversion)** to generate  
two difficulty levels: **hardest** and **harder** samples.

### 2-1️⃣ Clone the Seed-VC repository and set up its environment

```bash
cd data/preprocessing
git clone https://github.com/Plachtaa/seed-vc.git
```

Because Seed-VC uses different library versions (Hydra, PyTorch, etc.),  
create a separate environment named `seedvc`.

```bash
conda create -n seedvc python=3.10
conda activate seedvc
pip install -r seed-vc/requirements.txt
```

---

### 2-2️⃣ Install the ECAPA model dependency

The **speechbrain** package is required for ECAPA-TDNN-based speaker similarity.

```bash
pip install speechbrain
```

> 💡 The `speechbrain/spkrec-ecapa-voxceleb` model is used  
> to find the **most similar reference** for each speaker within VoxCeleb2  
> (for the *harder* case).

---

### 2-3️⃣ Add the Refined Inference Scripts

Copy the following two files into the `seed-vc` directory:

```bash
multi_inference_v2.py
multi_inference_v2.sh
```

> 🧩 These scripts automatically generate Refined RVFA:
> - **hardest** → self-reference case  
> - **harder** → ECAPA-selected reference case  
>
> Results will be saved in  
> `RealVideo-FakeAudio-Refine/hardest` and `RealVideo-FakeAudio-Refine/harder`.

---

### 2-4️⃣ Enter the Seed-VC directory

```bash
cd seed-vc
```

---

### 2-5️⃣ Run the Refined Inference

Use the provided SLURM script to process all files:

```bash
sbatch multi_inference_v2.sh
```

> ⚙️ Key arguments inside the script:
> - `SRC_ROOT`: Path to *RealVideo-RealAudio*  
> - `DST_ROOT`: Output path (*RealVideo-FakeAudio-Refine*)  
> - `VOX_ROOT`: VoxCeleb2 data path  
> - `--seed-vc-root`: Path to the seed-vc repository (relative or absolute)
>
> Example output:
> ```
> [DONE][GPU0] hardest=120 harder=120
> ================================
> hardest : 120
> harder  : 120
> =================================
> ```

After completion, both **harder** and **hardest** versions of Refined RVFA will be generated.

---

## 🎯 Step 3. Return to the AVDFD Environment

Deactivate the `seedvc` environment and return to `AVDFD`.

```bash
conda deactivate
conda activate AVDFD
```

---

## 🚀 Step 4. Training

You can now train your model using the generated data located in:

```
/RealVideo-FakeAudio-Refine/harder
/RealVideo-FakeAudio-Refine/hardest
```

Training scripts are available in the [`train/`](../train) directory.

---

## 📁 Directory Structure Example

```
FakeAVCeleb_v1.2/
├── RealVideo-RealAudio/
├── RealVideo-FakeAudio-Refine/
│   ├── harder/
│   │   ├── race/gender/id00001/clip001.mp4
│   │   └── ...
│   └── hardest/
│       ├── race/gender/id00001/clip001.mp4
│       └── ...
└── ...
```

---

## 🧠 Additional Information

| Item | Description |
|------|--------------|
| **VoxCeleb2 (Full)** | ≈ 436 GB (1,092,009 utterances / 6,112 speakers) |
| **Filtered VoxCeleb2** | ≤ 12 GB (only IDs overlapping with RVRA) |
| **SpeechBrain Model** | [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| **Seed-VC Repo** | [https://github.com/Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) |

---

## 🧩 Summary

| Step | Task | Environment |
|------|------|--------------|
| Step 0 | Environment setup (`AVDFD`) | conda |
| Step 1 | Download VoxCeleb2 subset | AVDFD |
| Step 2 | Generate Refined RVFA (Seed-VC) | seedvc |
| Step 3 | Switch back to AVDFD | conda |
| Step 4 | Model training | AVDFD |

---

> ✅ After completing these steps, you will have the **Refined RVFA dataset**  
> containing both *harder* and *hardest* voice-converted samples ready for training.