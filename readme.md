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
│   ├── file2.wav
│   └── ...
└── manifest/
```

---

## ⚙️ Preprocessing and Manifest Creation

### 1. Segment Audio

Use the segmentation script:

```bash
python segment_audio.py <input_dir> <segment_file> <output_dir>
```

Example `segment_file`:

```
seg1 audio1 0.0 5.0
seg2 audio2 10.0 15.0
```

This will generate `seg1.wav` and `seg2.wav`.

---

### 2. Prepare Manifest Files

```bash
python prepare_data.py /path/to/root_directory/
```

### Output Structure

```
manifest/
├── train.tsv
├── train.wrd
├── train.ltr
├── valid.tsv
├── valid.wrd
├── valid.ltr
├── test.tsv
├── test.wrd
├── test.ltr
└── dict.ltr.txt
```

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

Run the following command using `fairseq` and Hydra config:

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

## 🧪 Troubleshooting

- 🔍 **Zero-sample audio**: Remove or skip files with no samples to avoid crashing.
- 📉 **WER not improving?**
  - Adjust `learning_rate`, `update_freq`, or training duration.
  - Use a tri-stage LR scheduler (warmup → hold → decay).
  - Ensure transcript formatting is clean and consistent.

---

## 🔭 Future Work

- Add KenLM decoding and Hindi-specific LM integration.
- Try fine-tuning with larger multilingual datasets.
- Evaluate with WER across more benchmarks.

---

## 📜 License

This project inherits the [MIT License](https://choosealicense.com/licenses/mit/) from the original IndicWav2Vec repo.

---

## 📧 Contact

For queries or collaboration:

- Keyur Chaudhari – [keyur.email@example.com](mailto:keyur.email@example.com)
