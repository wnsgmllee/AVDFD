# Voice-Cloned Deepfake Detection with Refined Cross-Modal Hard Samples

Recent advances in voice cloning enable the creation of highly synchronized yet semantically fake audio, challenging traditional audio-visual deepfake detectors that rely mainly on lip–speech misalignment.
This project introduces a detection framework that explicitly targets voice-cloned deepfakes with near-perfect synchronization, focusing on subtle cross-modal inconsistencies rather than temporal mismatches.

To support this, we construct Refined RealVideo–FakeAudio (RVFA) — a hard, voice-cloned variant of the FakeAVCeleb dataset — where the original speech is replaced by personalized cloned voices while preserving the temporal structure.
This refined dataset serves as a challenging benchmark to push the boundary of next-generation deepfake detection methods.

---

## Step 0. Environment Setup (Preprocessing)

Move to the preprocessing directory and create the environment named `preprocessing`.

```bash
cd data/preprocessing
conda create -n preprocessing python=3.10 -y
conda activate preprocessing
```

### Recommended PyTorch version (for CUDA 12.x)

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```

Do not install additional requirements here — the rest will be handled later after cloning **Seed-VC**.

---

## Step 1. Dataset Download (VoxCeleb2 + FakeAVCeleb)

### 1-1️⃣ VoxCeleb2 Download (filtered by RVRA IDs)

Run the following script to download only the speaker IDs that exist in the RealVideo-RealAudio (RVRA) portion of FakeAVCeleb.  
This filtered download is less than 12 GB in total.

```bash
bash down_vox2.sh
```

✅ **Selective download mode:** The script automatically scans RVRA to collect all unique IDs and downloads only the corresponding VoxCeleb2 utterances for those speakers.

⚠️ **Full VoxCeleb2 download (optional)** — about 436 GB total:

```bash
bash voxceleb2_download.sh
```

If you choose the full download mode, install `hf_transfer` in your preprocessing environment:

```bash
conda install -c conda-forge hf_transfer -y
```

⚠️ **Warning:** The full VoxCeleb2 dataset is extremely large. Use the selective version unless you have sufficient SSD/NAS storage.

---

### 1-2️⃣ FakeAVCeleb Download

Download **FakeAVCeleb_v1.2** into the following directory:

```
data/FakeAVCeleb_Refine/
```

**How to download:**  
Please refer to the instructions in:
```
data/FakeAVCeleb_Refine/FakeAVCeleb_v1.2/README.md
```

⚠️ This dataset is required for generating the Refined RVFA dataset in later steps.

---

## Step 2. Voice Cloning (Seed-VC)

Refined RVFA is created using **Seed-VC: Towards Voice Conversion for All with Seed Learning**,  
a state-of-the-art open-source voice conversion model.  
Official implementation: [Seed-VC GitHub](https://github.com/Plachtaa/seed-vc)

This step runs entirely within the same preprocessing environment — no need to create a new one.

### 2-1️⃣ Clone the Seed-VC repository

```bash
cd data/preprocessing
git clone https://github.com/Plachtaa/seed-vc.git
```

### 2-2️⃣ Install Seed-VC dependencies

```bash
pip install -r seed-vc/requirements.txt
pip install datasets>=2.20.0 tqdm>=4.66.0
```

Additionally, install `speechbrain` for ECAPA-based similarity scoring:

```bash
pip install speechbrain
```

💡 The `speechbrain/spkrec-ecapa-voxceleb` model is used to find the most similar reference for each speaker within VoxCeleb2.

### 2-3️⃣ Move Inference Scripts

```bash
mv data/preprocessing/multi_inference_v2.py seed-vc/
mv data/preprocessing/multi_inference_v2.sh seed-vc/
```

These scripts automatically generate Refined RVFA:

- **hardest** → self-reference case  
- **harder** → ECAPA-selected reference case  

Results will be saved in `RealVideo-FakeAudio-Refine/hardest` and `RealVideo-FakeAudio-Refine/harder`.

### 2-4️⃣ Enter the Seed-VC directory

```bash
cd seed-vc
```

### 2-5️⃣ Run the Refined Inference

Use the provided SLURM script to process all files:

```bash
sbatch multi_inference_v2.sh
```

**Key arguments inside the script:**
- `SRC_ROOT`: Path to RealVideo-RealAudio  
- `DST_ROOT`: Output path (RealVideo-FakeAudio-Refine)  
- `VOX_ROOT`: VoxCeleb2 data path  
- `--seed-vc-root`: Path to the seed-vc repository

Example output:

```bash
[DONE][GPU0] hardest=120 harder=120
================================
hardest : 120
harder  : 120
================================
```

After completion, both harder and hardest versions of Refined RVFA will be generated.

---

## Step 3. Main Environment Setup (AVDFD)

Now we create the main training environment named `AVDFD`.  
We use a separate environment because **Seed-VC** and **OpenAVFF** have conflicting dependencies.

The baseline is based on **AVFF: Audio-Visual Forgery Face Dataset (CVPR 2024)**,  
implemented using [OpenAVFF (GitHub)](https://github.com/JoeLeelyf/OpenAVFF).

### 3-1️⃣ Create the AVDFD environment

```bash
conda create -n AVDFD python=3.10 -y
conda activate AVDFD
```

Install the following packages:

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r OpenAVFF/requirements.txt
```

### 3-2️⃣ Download and Prepare Stage2 Weights

Download **Stage2 Weights** from the [OpenAVFF repository](https://github.com/JoeLeelyf/OpenAVFF) and save them to `OpenAVFF/checkpoints/`.

### 3-3️⃣ Convert Stage2 to Stage3 Initialization

Inside the `OpenAVFF` folder, run the following code:

```bash
python convert_stage2_to_stage3.py --stage2 checkpoints/stage2_pretrained.pth -out checkpoints/stage3_init_from_stage2.pth --num_classes 2
```

---

## Step 4. Training

You can now train your model using the generated data located in:

```bash
/RealVideo-FakeAudio-Refine/harder
/RealVideo-FakeAudio-Refine/hardest
```

Training scripts are available in the `train/` directory.

---

## 📁 Dataset Directory Structure Example

```bash
FakeAVCeleb_v1.2/
├── RealVideo-RealAudio/
├── RealVideo-FakeAudio-Refine/
│   ├── harder/
│   │   ├── race/gender/id00001/clip001.mp4
│   │   └── ...
│   └── hardest/
│       ├── race/gender/id00001/clip001.mp4
│       └── ...
│
```
