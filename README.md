# BERT Hacking

A minimal, configurable pipeline for fine-tuning BERT (and other Hugging Face transformers) on text classification tasks.

This project focuses on **simplicity, reproducibility, and fast experimentation** and is based on Hugging Face classes.

---

## 🚀 Features

- Train any Hugging Face model (`bert`, `distilbert`, `roberta`, etc.)
- YAML-based configuration (no code changes needed)
- Clean preprocessing pipeline
- Train / validation / test split
- Built-in evaluation metrics (accuracy, precision, recall, F1)
- Automatic experiment hashing for reproducibility
- Batch prediction with probability scores

---

## 📂 Project Structure

    .
    ├── config.yml          # Experiment configuration
    ├── train.py            # Main entry point
    ├── model.py            # Training + inference logic
    ├── preprocess.py       # Data cleaning and formatting
    ├── evaluate.py         # Metrics
    ├── splits.py           # Dataset splitting
    ├── experiment.py       # Experiment hashing
    ├── environment.py      # Environment setup / cleanup
    └── config.py           # Config loader

---

## ⚙️ Installation

    pip install -r requirements.txt

Or manually:

    pip install torch transformers datasets pandas scikit-learn pyyaml

---

## 🧪 Usage

### 1. Prepare your dataset

CSV format example:

    id,text,label
    1,"This is great!",1
    2,"Terrible experience",0

---

### 2. Configure your experiment

Edit `config.yml`:

    data:
      input_path: data/train.csv
      text_column: text
      label_column: label
      id_column: id

    split:
      train_size: 0.8
      validation_size: 0.1
      test_size: 0.1
      stratify: true

    model:
      model_name: distilbert-base-uncased
      num_labels: 2

    training:
      output_dir: outputs/run_001
      learning_rate: 2e-5
      per_device_train_batch_size: 8
      num_train_epochs: 3

---

### 3. Run training

    python train.py --config config.yml

(or just `python train.py` if using default config path)

---

## 📊 Output

After training, you’ll get:

- 📁 Model checkpoints → `output_dir`
- 📄 Test predictions →  
  `test_predictions_<run_hash>.csv`

Example output:

    id,prediction,scores,true_label
    1,1,"[0.1, 0.9]",1

---

## How it works

### Pipeline

1. Load dataset from CSV  
2. Sanitize columns → `TEXT`, `LABEL`, `ID`  
3. Split into train / validation / test  
4. Tokenize using Hugging Face tokenizer  
5. Train using `Trainer`  
6. Evaluate with standard metrics  
7. Generate predictions  

---

## Configuration Explained

### `data`
- Defines dataset location and column names

### `split`
- Controls dataset splitting and reproducibility

### `model`
- Hugging Face model + number of classes

### `training`
- Hyperparameters passed to `TrainingArguments`

---

## Reproducibility

Each run generates a unique hash based on:

- model  
- split ratios  
- learning rate  
- batch size  
- epochs  
- seed  

This ensures experiments are easy to track and compare.

---

## Customization

You can:

- Swap models:
  
      model_name: roberta-base

- Change task type:
  - binary classification
  - multiclass classification

- Adjust training:
  
      learning_rate: 3e-5
      num_train_epochs: 5

---

## ⚠️ Limitations

This project is intentionally minimal. It does **not** include:

- hyperparameter tuning  
- experiment tracking (e.g. WandB)  
- distributed training  
- advanced callbacks  
- custom architectures  

---
