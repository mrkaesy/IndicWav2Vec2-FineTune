# IndicWav2Vec Fine-Tuning on Hindi

This project walks through the complete process of fine-tuning the [IndicWav2Vec Base model](https://github.com/AI4Bharat/IndicWav2Vec) for Automatic Speech Recognition (ASR) using the [SPRING-INX Hindi dataset](https://asr.iitm.ac.in/dataset). You will learn how to prepare your dataset, preprocess audio, create manifests, fine-tune the model, and evaluate performance.

---

## 📁 Dataset Overview

- **Source**: Spring Lab (IIT Madras)
- **Language**: Hindi
- **Total Hours**: 351.18 hours
  - Train: 316.41 hours
  - Validation: 29.68 hours
  - Test: 5.09 hours

---

## 🛠️ Environment Setup

### Step 1: Create Conda Environment

```bash
conda create -n wav2vec-finetune python=3.9
conda activate wav2vec-finetune
```

### Step 2: Install System Dependencies

```bash
sudo apt-get install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev build-essential cmake libboost-all-dev libeigen3-dev ffmpeg
```

### Step 3: Install Python Requirements

```bash
pip install -r w2v_inference/requirements.txt
pip install packaging soundfile swifter editdistance omegaconf pandas
```

### Step 4: Install Fairseq (Customized)

```bash
git clone https://github.com/AI4Bharat/fairseq.git
cd fairseq
pip install --editable ./
```

---

## 📦 Data Preparation Pipeline

All scripts are located in [`data_prep_scripts`](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/data_prep_scripts).

### Step-by-Step Data Preparation

You can either use individual scripts or run everything via the master script `process_data.sh`.

### 🔽 1. Download Data

Prepare a text file `urls.txt` with one line per audio file URL.

```bash
bash dw_util.sh urls.txt /data/audio_downloads 4
```

- `urls.txt`: File containing download URLs
- `/data/audio_downloads`: Directory to store the downloaded files
- `4`: Number of parallel download threads

---

### 🔇 2. Voice Activity Detection (VAD)

Removes silent portions of the audio.

```bash
python vad.py /data/audio_downloads /data/vad_output hindi
```

- `/data/audio_downloads`: Root directory containing language folders
- `/data/vad_output`: Destination directory for VAD-processed audio
- `hindi`: Folder name to process

---

### 🔊 3. SNR Filtering

Filters audio segments with poor signal-to-noise ratio.

```bash
python snr_filter.py /data/vad_output hindi
```

Filtered files are stored, and noisy ones are moved to `snr_rejected/`.

---

### ✂️ 4. Chunking

Splits long audio into manageable 15s segments.

```bash
python chunking.py /data/vad_output/hindi
```

This replaces original files with chunked ones in-place.

---

### 🚀 OR Run All Steps in One Command

```bash
bash process_data.sh /data/audio_downloads 4
```

---

## 🧾 Manifest Creation

### Language-wise Manifest

```bash
python lang_wise_manifest_creation.py /data/wav_files --dest /data/manifest --ext wav --valid-percent 0.03
```

- `--ext`: File extension (e.g., `wav`)
- `--valid-percent`: Portion of training data to use for validation

### Combine Validation Files

```python
import pandas as pd
import glob

files = glob.glob("*_valid.tsv")
dfs = [pd.read_csv(f, skiprows=1, names=['f', 'd'], sep='\t') for f in files]
combined = pd.concat(dfs)
combined.to_csv('valid.tsv', index=False, header=False, sep='\t')
```

Add root directory path as the first line of `valid.tsv`.

---

## 🧠 Fine-Tuning Configuration

| Parameter             | Value                                                   |
|-----------------------|---------------------------------------------------------|
| Model                 | IndicWav2Vec Base                                       |
| Loss Function         | CTC (Connectionist Temporal Classification)             |
| Optimizer             | Adam                                                    |
| Learning Rate         | `0.00001`                                               |
| Update Frequency      | `4`                                                     |
| Max Updates           | `1,520,000`                                             |
| Pretrained Checkpoint | [Download](https://indic-asr-public.objectstore.e2enetworks.net/aaai_ckpts/pretrained_models/indicw2v_base_pretrained.pt) |

---

## 🧪 Fine-Tuning Command

```bash
fairseq-hydra-train \
  task.data=/data/manifest \
  model.w2v_path=/models/indicw2v_base_pretrained.pt \
  common.log_interval=50 \
  dataset.max_tokens=1000000 \
  checkpoint.save_dir=/models/checkpoints_hindi \
  +optimization.update_freq='[4]' \
  optimization.lr=0.00001 \
  distributed_training.distributed_world_size=1 \
  --config-dir /configs \
  --config-name ai4b_base
```

---

## 📊 Results

| Model              | WER (Validation) | WER (Test) | Updates     |
|--------------------|------------------|------------|-------------|
| IndicWav2Vec Base  | 29.98            | 29.93      | 1,520,000   |

---

## 🔭 Language Model Training

Refer to [`lm_training`](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/lm_training) to install and train 6-gram KenLM language models.

---

## 📈 Evaluation and Inference

See [`w2v_inference`](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/w2v_inference) for decoding and evaluation scripts.

```bash
python infer.py --path model.pt --task audio_finetuning --gen-subset test --results-path results/ ...
```

---

## 🧪 Troubleshooting Tips

- ❌ **Zero-length files**: Skip them to prevent crashes.
- 📉 **WER not improving**: Check learning rate, transcript alignment, and try using a tri-stage LR schedule.
- 🗂️ **Data issues**: Ensure manifest paths are valid and match audio files.

---

## 📜 License

This project uses the [MIT License](https://choosealicense.com/licenses/mit/).

---

## 📖 Paper & Citation

- [arXiv: Towards Building ASR Systems for the Next Billion Users](https://arxiv.org/abs/2111.03945)

```bibtex
@inproceedings{javed2021building,
    title = {Towards Building ASR Systems for the Next Billion Users},
    author = {Tahir Javed and Sumanth Doddapaneni and Abhigyan Raman and Kaushal Santosh Bhogale and Gowtham Ramesh and Anoop Kunchukuttan and Pratyush Kumar and Mitesh M. Khapra},
    booktitle = "Proceedings of the AAAI Conference on Artificial Intelligence",
    year = "2022",
}
```

---

## 📧 Contact

- Keyur Chaudhari – [keyur.email@example.com](mailto:keyur.email@example.com)
