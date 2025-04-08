# IndicWav2Vec Fine-Tuning on Hindi

This project demonstrates fine-tuning the [IndicWav2Vec Base model](https://github.com/AI4Bharat/IndicWav2Vec) for Automatic Speech Recognition (ASR) using the [SPRING-INX Hindi dataset](https://asr.iitm.ac.in/dataset). The final model achieves a Word Error Rate (WER) of **29.98** on the test set.

---

## 📁 Dataset

- **Dataset**: Spring Lab (IIT Madras) Hindi ASR Dataset  
- **Total Hours**: 351.18 hours  
- **Split**:
  - Train: 316.41 hours
  - Valid: 29.68 hours
  - Test: 5.09 hours

The dataset must be segmented and normalized to 16kHz WAV format. The directory structure should look like this:

```
/root_directory/
├── hindi/
│   ├── transcript.txt
│   ├── file1.wav
│   └── ...
└── manifest/
```

---

## 🛠️ Environment Setup

### 1. Create a Conda Environment

```bash
conda create -n wav2vec-finetune python=3.9
conda activate wav2vec-finetune
```

### 2. Install System Dependencies

```bash
sudo apt-get install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev \
build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev ffmpeg
```

### 3. Install Python Dependencies

```bash
pip install -r w2v_inference/requirements.txt
pip install packaging soundfile swifter editdistance omegaconf
```

### 4. Install Fairseq

```bash
git clone https://github.com/AI4Bharat/fairseq.git
cd fairseq
pip install --editable ./
```

---

## ⚙️ Data Preparation

All data preparation scripts are located in the [`data_prep_scripts`](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/data_prep_scripts) directory.

The pipeline includes:
1. **Downloading the data**
2. **Voice Activity Detection (VAD)** — `vad.py`
3. **SNR Filtering** — `snr_filter.py`
4. **Audio Chunking** — `chunking.py`

These steps are automatically handled by:

```bash
bash process_data.sh <path_to_urls> <data_store_path> <num_threads>
```

---

## 🧾 Manifest Creation

Create language-specific manifests:

```bash
python path/to/lang_wise_manifest_creation.py /path/to/wave/files --dest /manifest/path --ext wav --valid-percent 0.03
```

To generate a combined validation manifest:

```python
import pandas as pd
import glob

filenames = glob.glob("*_valid.tsv")
combined = []

for f in filenames:
    df = pd.read_csv(f, skiprows=1, names=['f', 'd'], sep='\t')
    combined.append(df)

df_combined = pd.concat(combined, axis=0, ignore_index=True)
df_combined.to_csv('valid.tsv', index=True, header=False, sep='\t')
```

Ensure `/path/to/wav/files/` is added to the first line of `valid.tsv`.

---

## 🧠 Fine-Tuning Configuration

| Parameter             | Value                                                   |
|-----------------------|---------------------------------------------------------|
| Model                 | IndicWav2Vec Base                                       |
| Loss Function         | Connectionist Temporal Classification (CTC)             |
| Optimizer             | Adam                                                    |
| Learning Rate         | `0.00001`                                               |
| Update Frequency      | `4`                                                     |
| Max Updates           | `1,520,000`                                             |
| Pretrained Checkpoint | [Download](https://indic-asr-public.objectstore.e2enetworks.net/aaai_ckpts/pretrained_models/indicw2v_base_pretrained.pt) |

---

## 🚀 Fine-Tuning Command

```bash
fairseq-hydra-train \
  task.data=/path/to/manifest/ \
  model.w2v_path=/path/to/indicw2v_base_pretrained.pt \
  common.log_interval=50 \
  dataset.max_tokens=1000000 \
  checkpoint.save_dir=/path/to/save_checkpoints/ \
  +optimization.update_freq='[4]' \
  optimization.lr=0.00001 \
  distributed_training.distributed_world_size=1 \
  --config-dir /path/to/configs/ \
  --config-name ai4b_base
```

---

## 📊 Results

| Model              | WER (Validation) | WER (Test) | Updates     |
|--------------------|------------------|------------|-------------|
| IndicWav2Vec Base  | 29.98            | 29.93      | 1,520,000   |

---

## 📚 Training Language Model

Scripts for preparing and training LMs are in the [`lm_training`](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/lm_training) directory.

---

## 🧪 Inference and Evaluation

Scripts and documentation for inference and evaluation are in the [`w2v_inference`](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/w2v_inference) directory.

---

## 🔭 Future Work

- Add KenLM decoding and Hindi-specific LM integration.
- Fine-tune on larger multilingual corpora.
- Optimize tri-stage learning rate for better convergence.

---

## 🧪 Troubleshooting

- 🔍 **Zero-sample audio**: Remove or skip to prevent crashes.
- 📉 **WER stagnant**:
  - Tune learning rate or update frequency
  - Check manifest and data consistency
  - Use a tri-stage learning rate schedule

---

## 📜 License

IndicWav2Vec is [MIT-licensed](https://choosealicense.com/licenses/mit/). Applies to pretrained, fine-tuned, and language models.

---

## 📖 Paper & Citation

- arXiv: [Towards Building ASR Systems for the Next Billion Users](https://arxiv.org/abs/2111.03945)

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

For queries or collaboration:

- Keyur Chaudhari – [keyur.email@example.com](mailto:keyur.email@example.com)
