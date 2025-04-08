# IndicWav2Vec2 Fine-Tuning for Hindi ASR

This guide provides an extensive step-by-step walkthrough of fine-tuning the [IndicWav2Vec2](https://github.com/AI4Bharat/IndicWav2Vec) Base model for Hindi speech recognition using the [SPRING-INX Dataset](https://asr.iitm.ac.in/dataset). It includes detailed setup instructions, data preparation, model fine-tuning, evaluation, and optimization strategies.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Data Preparation Scripts Explained](#data-preparation-scripts-explained)
- [Manifest Creation](#manifest-creation)
- [Model Overview](#model-overview)
- [Fine-Tuning Pipeline](#fine-tuning-pipeline)
- [Evaluation](#evaluation)
- [Results and Observations](#results-and-observations)
- [Future Work](#future-work)
- [References](#references)

---

## 📢 Introduction

We explore the fine-tuning of IndicWav2Vec2, a speech recognition model pretrained on 40 Indian languages, for Hindi Automatic Speech Recognition (ASR). We demonstrate the process with the Spring Lab IITM Hindi dataset (~351 hrs), achieving a final Word Error Rate (WER) of **29.93**. The work also explores optimization strategies like learning rate schedules and CTC loss.

---

## 🛠️ Environment Setup

```bash
conda create -n wav2vec-finetune python=3.9
conda activate wav2vec-finetune
sudo apt-get install libsndfile1-dev ffmpeg cmake libboost-all-dev libeigen3-dev
pip install -r w2v_inference/requirements.txt
pip install packaging soundfile swifter editdistance omegaconf
```

### Install Fairseq

```bash
git clone https://github.com/AI4Bharat/fairseq.git
cd fairseq
pip install --editable ./
```

---

## 📚 Dataset and Preprocessing

- **Dataset**: Spring Lab IITM - Hindi (351.18 hrs)
- **Structure**: Includes large unsegmented audio files with transcript
- **Goal**: Prepare data with ≤15s WAV segments and generate manifest

---

## 🧾 Data Preparation Scripts Explained

### 1. `dw_util.sh`
**Downloads audio files from URLs.**

```bash
bash dw_util.sh urls.txt /data/downloads 4
```

---

### 2. `vad.py`
**Removes silent portions from the audio using Voice Activity Detection.**

```bash
python vad.py /data/downloads /data/vad_output hindi
```

---

### 3. `snr_filter.py`
**Removes noisy audio using Signal-to-Noise Ratio (SNR) filtering.**

```bash
python snr_filter.py /data/vad_output hindi
```

---

### 4. `chunking.py`
**Splits long audio files into ≤15s chunks.**

```bash
python chunking.py /data/vad_output/hindi
```

---

### OR Run All in One

```bash
bash process_data.sh /data/downloads 4
```

---

## 📁 Manifest Creation

### Directory Structure

```
/data/
├── hindi/
│   ├── file1.wav
│   ├── transcript.txt
└── manifest/
```

### Script

```bash
python lang_wise_manifest_creation.py /data/hindi --dest /data/manifest --ext wav --valid-percent 0.03
```

Output:
- `train.tsv`, `train.wrd`, `train.ltr`
- `valid.tsv`, ...
- `test.tsv`, ...
- `dict.ltr.txt`

---

## 🧠 Model Overview

### IndicWav2Vec Base

| Property         | Value     |
|------------------|-----------|
| Transformer      | 12 layers |
| Hidden Units     | 768       |
| Params           | 95M       |
| Optimizer        | Adam      |
| Loss             | CTC       |

### Architecture:
- Feature Encoder (CNN)
- Context Network (Transformer)
- Quantization (pretraining only)

---

## 🏁 Fine-Tuning Pipeline

### Training Command

```bash
fairseq-hydra-train \
  task.data=/data/manifest \
  model.w2v_path=indicw2v_base_pretrained.pt \
  checkpoint.save_dir=/checkpoints/ \
  +optimization.update_freq='[4]' \
  optimization.lr=0.00001 \
  distributed_training.distributed_world_size=1 \
  --config-dir /configs \
  --config-name ai4b_base
```

---

## 📈 Evaluation

### Inference Script

```bash
python infer.py \
  --path checkpoints/model.pt \
  --task audio_finetuning \
  --gen-subset test \
  --results-path /results
```

---

## 📊 Results and Observations

| Model                 | WER (Validation) | WER (Test) | Updates     |
|-----------------------|------------------|------------|-------------|
| IndicWav2Vec2.0 Base  | 29.98            | 29.93      | 1,520,000   |
| Data2Vec Base         | 27.98            | 28.36      | 479,000     |

### Learning Rate Tuning (Tri-stage)

| LR      | Update Freq | WER   | Time        |
|---------|-------------|-------|-------------|
| 0.0001  | 1           | 33.71 | 2d 18h      |
| 0.00003 | 4           | 28.36 | 5d 17h      |

**WER stagnation** observed at 33.71 was resolved by:
- Removing zero-sample audio
- Gradual tuning of learning rate

---

## 🚀 Future Work

- LoRA-based fine-tuning for efficient adaptation
- Inference API with form-assist bot (OCR + GPT + ASR)
- Domain-specific LM integration
- Code-switched dataset: Hinglish

---

## 📚 References

1. [AI4Bharat IndicWav2Vec](https://github.com/AI4Bharat/IndicWav2Vec)
2. [Illustrated Wav2Vec2.0](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)
3. [SPRING-INX Hindi Dataset](https://asr.iitm.ac.in/dataset)
4. [Wav2Vec2.0 Paper](https://arxiv.org/abs/2006.11477)

---

For more technical documentation and experiments, refer to:
- `readme1.md`, `README2.md`, and `README3.md` (combined here)
- Mid-term and final reports (included above)

