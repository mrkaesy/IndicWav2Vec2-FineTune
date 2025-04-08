
# 🔊 IndicWav2Vec2 — End‑to‑End Fine‑Tuning Guide (Hindi ASR)
> **Goal:** take raw Hindi audio + transcripts ➜ produce a state‑of‑the‑art Automatic Speech Recognition (ASR) model using the **IndicWav2Vec2 Base** checkpoint.

---

## 0 · Why This Guide Exists
Most academic READMEs assume you already know:
* what *manifest files* are,
* how to convert MP3 → WAV,
* what `fairseq-hydra-train` even does…

**This document is written for absolute newcomers** who want to reproduce our results from scratch on **any Linux box with ≥1 GPU**.

---

## 1 · Project Layout at a Glance

```
indicwav2vec_finetune/
├── data_prep_scripts/        # All audio‑processing helpers (download, VAD, SNR…)
│   ├── dw_util.sh
│   ├── vad.py
│   ├── snr_filter.py
│   ├── chunking.py
│   ├── process_data.sh
│   └── lang_wise_manifest_creation.py
├── configs/                  # Pre‑made Hydra configs for pre‑training
├── finetune_configs/         # Hydra configs for fine‑tuning
├── lm_training/              # KenLM installation + training scripts
├── w2v_inference/            # Inference / evaluation helpers
├── reports/                  # PDF reports (mid‑term, final)
└── README_ULTIMATE.md        # ← **YOU ARE HERE**
```

> **Tip:** keep this repo at `~/indicwav2vec_finetune` so the paths below match exactly.

---

## 2 · Prerequisites

| Software            | Version (tested)      | Install command |
|---------------------|-----------------------|-----------------|
| Ubuntu              | 20.04 LTS             | — |
| CUDA Toolkit        | 11.7                  | see NVIDIA docs |
| Conda (Miniconda)   | ≥23.x                 | <https://conda.io> |
| Git                 | 2.34+                 | `sudo apt install git` |

### 2.1 Create and Activate Environment

```bash
conda create -n wav2vec-finetune python=3.9 -y
conda activate wav2vec-finetune
```

### 2.2 System Packages (audio libs, BLAS, etc.)

```bash
sudo apt-get update
sudo apt-get install -y libsndfile1-dev ffmpeg      liblzma-dev libbz2-dev libzstd-dev      build-essential cmake libboost-all-dev libeigen3-dev
```

### 2.3 Python Dependencies

```bash
# clone our repo first
git clone https://github.com/yourname/indicwav2vec_finetune.git
cd indicwav2vec_finetune

pip install -r w2v_inference/requirements.txt   # torchaudio, hydra, etc.
pip install packaging soundfile swifter editdistance omegaconf pandas
```

### 2.4 Fairseq (custom fork used by AI4Bharat)

```bash
git clone https://github.com/AI4Bharat/fairseq.git
cd fairseq
pip install --editable .
cd ..
```

You are now ready to process data.

---

## 3 · Dataset Acquisition

### 3.1 SPRING‑INX Hindi

* **Hours:** 351.18  
* **Where:** <https://asr.iitm.ac.in/dataset>

Create a file `urls.txt` containing **direct links** to each WAV/ZIP provided by the site.

```
datasets/
└── urls.txt   # one URL per line
```

---

## 4 · Data Preparation (4‑stage pipeline)

All helper scripts live in **`data_prep_scripts/`**.

| Stage | Script | What it does | Input → Output |
|-------|--------|--------------|----------------|
| 1 | `dw_util.sh` | Parallel downloader | `urls.txt` → raw WAV/ZIP |
| 2 | `vad.py` | Removes long silences | raw WAV → `*_vad.wav` |
| 3 | `snr_filter.py` | Drops low‑quality audio (SNR < 20 dB) | `*_vad.wav` → clean WAV |
| 4 | `chunking.py` | Splits >15 s files into chunks | clean WAV → ≤15 s WAV |

### 4.1 Run Everything in One Command

```bash
bash data_prep_scripts/process_data.sh datasets/urls.txt      /mnt/hindi_audio 8
```

* **`/mnt/hindi_audio`** will end up like:

```
/mnt/hindi_audio/
├── hindi/
│   ├── 000001.wav
│   ├── 000002.wav
│   └── ...
└── rejected/           # noisy files here
```

> **FAQ:** *Where are transcripts?*  
> Put each language’s `transcript.txt` next to its WAVs **before** you run stage 4; the chunker keeps filenames intact.

### 4.2 What If I Only Want VAD?

```bash
python data_prep_scripts/vad.py /mnt/raw /mnt/vad hindi
```

---

## 5 · Manifest Creation (Fairseq format)

### 5.1 Generate Language‑Wise TSVs

```bash
python data_prep_scripts/lang_wise_manifest_creation.py        /mnt/hindi_audio/hindi        --dest /mnt/hindi_manifest        --ext wav --valid-percent 0.03
```

You’ll get:

```
/mnt/hindi_manifest/
├── train.tsv   # path 	 duration(ms)
├── train.wrd   # words per line
├── train.ltr   # letters per line
└── ... (valid / test)
```

### 5.2 Combine Multiple Languages (optional)

If you ever pretrain, concatenate all `*_valid.tsv` then prepend root path as first line:

```python
import glob, pandas as pd, pathlib, sys
root = pathlib.Path("/mnt/hindi_audio")
dfs = [pd.read_csv(f, sep='\t', names=['path','dur'], skiprows=1)
       for f in glob.glob("*_valid.tsv")]
pd.concat(dfs).to_csv("valid.tsv", sep='\t', header=False, index=False)
```

---

## 6 · Model Overview

| Checkpoint | Link | Size |
|------------|------|------|
| **Base**   | [`indicw2v_base_pretrained.pt`](https://indic-asr-public.objectstore.e2enetworks.net/aaai_ckpts/pretrained_models/indicw2v_base_pretrained.pt) | 95 M params |
| **Large**  | `indicw2v_large_pretrained.pt` | 317 M params |

We’ll fine‑tune **Base**.

---

## 7 · Fine‑Tuning

### 7.1 Minimal Single‑GPU Command

```bash
fairseq-hydra-train   task.data=/mnt/hindi_manifest   model.w2v_path=indicw2v_base_pretrained.pt   checkpoint.save_dir=/mnt/checkpoints_hindi   +optimization.update_freq='[4]'   optimization.lr=0.00001   dataset.max_tokens=1000000   common.log_interval=50   --config-dir finetune_configs   --config-name ai4b_base
```

### 7.2 Multi‑Node (Slurm) Template

See `README_SUPER_DETAILED.md` for full `sbatch` line, or reuse:

```bash
sbatch --job-name hindi_ft --gres gpu:4 --nodes 1 --cpus-per-task 16   --wrap "srun fairseq-hydra-train ... "
```

---

## 8 · Evaluation

```bash
python w2v_inference/infer.py    /mnt/hindi_manifest    --path /mnt/checkpoints_hindi/checkpoint_best.pt    --task audio_finetuning    --gen-subset test    --results-path results_hindi    --w2l-decoder viterbi
```

`results_hindi/wer` will show per‑sample and average WER.

---

## 9 · Common Pitfalls & Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `RuntimeError: Expected 1D tensor, got 0D` | Empty WAV (0 samples) | Remove file or re‑run `snr_filter.py` |
| WER stuck at 33 % | LR too high or update_freq=1 | use `0.00001` and `update_freq 4` |
| CUDA OOM | `max_tokens` too big | lower to `500000` |

---

## 10 · Training a Language Model (KenLM)

```bash
cd lm_training
bash install_kenlm.sh           # one‑time build
bash train_lm.sh /mnt/hindi_manifest hindi
```

Produces `lm.binary` and `lexicon.lst` usable with `--w2l-decoder kenlm`.

---

## 11 · Advanced: LoRA Fine‑Tuning (coming soon)

We plan to inject Low‑Rank Adaptation layers into every Transformer block to cut GPU RAM by 60 %. Stay tuned.

---

## 12 · Results Snapshot

| Config | max_update | LR | Update Freq | Best WER |
|--------|------------|----|-------------|----------|
| C1     | 1.72 M     | 1e‑4 | 1 | 33.71 |
| **C2** | **1.52 M** | **1e‑5** | **4** | **29.93** |

---

## 13 · References

See the **reports/** folder for full academic write‑ups and bibliography.

---

Happy fine‑tuning! If anything is unclear, open an issue or email **Keyur Chaudhari** at _keyur.email@example.com_.
