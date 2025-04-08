# Fine-Tuning the IndicWav2Vec2 Model for Hindi ASR

This tutorial provides a comprehensive guide on fine-tuning the IndicWav2Vec2 Base model for Automatic Speech Recognition (ASR) on Hindi data using the Spring Lab IIT Madras dataset. The process includes dataset preparation, preprocessing, manifest creation, model understanding, fine-tuning, evaluation, and recommendations for improvement.

---

## Part 1: Dataset Preparation

### 1.1 Dataset Source

**Dataset:** Spring-INX Hindi Dataset from IIT Madras  
**Total Hours:** 351.18 hours  
**Train/Validation/Test Split:**
- Train: 316.41 hours
- Validation: 29.68 hours
- Test: 5.09 hours

**Download URL:** [https://asr.iitm.ac.in/dataset](https://asr.iitm.ac.in/dataset)

### 1.2 Directory Structure

Ensure your dataset directory follows this structure:

```bash
/SPRING_INX_Hindi_R1/
├── audio/
│   ├── hi_IN_*.wav
├── transcript.txt
├── train/
├── valid/
├── test/
├── segments
├── spk2utt
├── utt2spk
├── utt2dur
├── text
├── wav.scp
```

Each audio file should:
- Be in `.wav` format
- Have a maximum duration of 15 seconds
- Be sampled at 16kHz

---

## Part 2: Data Preprocessing

### 2.1 Audio Segmentation

Use a Python script to segment long audio files based on timestamps.

**Command Format:**
```bash
python segment_audio.py <input_dir> <segment_file> <output_dir>
```

**Example segment file:**
```
seg1 audio1 0.0 5.0
seg2 audio2 10.0 15.0
```

This will produce:
- `seg1.wav`: First 5s of `audio1.wav`
- `seg2.wav`: From 10s to 15s of `audio2.wav`

### 2.2 Normalize & Create Manifest

Use the AI4Bharat preprocessing script to generate required manifest files.

**Command Format:**
```bash
python prepare_manifest.py /path/to/root_directory
```

**Directory Structure:**
```bash
/root_directory/
├── hindi/
│   ├── transcript.txt
│   ├── file1.wav
│   ├── file2.wav
├── manifest/
│   ├── train.tsv
│   ├── train.wrd
│   ├── train.ltr
│   ├── valid.tsv
│   ├── valid.wrd
│   ├── valid.ltr
│   ├── test.tsv
│   ├── test.wrd
│   ├── test.ltr
│   └── dict.ltr.txt
```

---

## Part 3: Model Architecture

### 3.1 IndicWav2Vec2 Overview

IndicWav2Vec2 is based on the Wav2Vec2.0 architecture and pre-trained on 17,314 hours of multilingual Indian speech data.

### 3.2 Architecture Comparison

| Feature             | Base             | Large           |
|---------------------|------------------|------------------|
| Transformer Layers  | 12               | 24               |
| Hidden Units        | 768              | 1024             |
| Attention Heads     | 8                | 16               |
| Parameters          | ~95M             | ~317M            |
| Speed               | Faster           | Slower           |

---

## Part 4: Fine-Tuning Procedure

### 4.1 Training Command Example

Use Fairseq's `fairseq-hydra-train` command:

```bash
fairseq-hydra-train \
  task.data=/path/to/manifest \
  model.w2v_path=/path/to/indicwav2vec.pt \
  model.freeze_finetune_updates=0 \
  optimization.lr=0.00005 \
  optimization.max_update=1520000 \
  optimization.update_freq=[4] \
  --config-dir /your/config/path \
  --config-name finetune.yaml
```

### 4.2 Key Hyperparameters

| Parameter                 | Description                                   |
|---------------------------|-----------------------------------------------|
| `max_update`              | Maximum number of updates                     |
| `lr`                      | Learning rate                                 |
| `update_freq`             | Gradient accumulation steps                   |
| `freeze_finetune_updates` | Set to 0 to train the full model              |

### 4.3 Learning Rate Strategy
- **Warmup:** First 10% of steps
- **Hold:** Next 40% of steps
- **Decay:** Remaining 50% (exponential)

---

## Part 5: Training Loss - CTC

### 5.1 CTC Loss Overview
CTC loss allows alignment between input (audio features) and output (text) without requiring exact frame-by-frame alignment.

Example: Speech input "hello" → aligns to character outputs 'h', 'e', 'l', 'l', 'o' over time.

---

## Part 6: Evaluation

### 6.1 Results Summary

| Learning Rate | Update Freq | Best WER | Finetune Time       |
|---------------|-------------|----------|---------------------|
| 0.0001        | 1           | 33.71    | 2 days, 18 hours    |
| 0.00001       | 4           | 29.98    | 7 days, 12 hours    |
| 0.00003       | 4           | 28.36    | 5 days, 17 hours    |

### 6.2 Performance on SPRING Test Set

| Model               | WER    |
|---------------------|--------|
| data2vec-aqc L      | 28.3   |
| IndicWav2Vec2.0 L   | 35.4   |
| IndicWav2Vec2.0 B   | 29.93  |

---

## Part 7: Troubleshooting Tips

| Issue                            | Solution                                      |
|----------------------------------|-----------------------------------------------|
| WER stagnation                   | Tune learning rate and update_freq            |
| Kernel/input size mismatch       | Ensure correct audio format (16kHz, ≤15s)     |
| Audio files with zero samples    | Filter such samples before training           |
| Dict mismatch errors             | Regenerate `dict.ltr.txt` including all chars |

---

## Part 8: Future Work

- Integrate **LoRA** for efficient parameter adaptation
- Fine-tune on **code-switched data** (e.g. Hinglish)
- Use **domain-specific language models** to improve decoding
- Deploy real-time ASR API for field testing

---

## Resources
- IndicWav2Vec: https://github.com/AI4Bharat/IndicWav2Vec
- Dataset: https://asr.iitm.ac.in/dataset
- Wav2Vec2.0 Explained: https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html

---

Let me know if you'd like the YAML config or training scripts.

