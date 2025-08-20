# TextSummarizer USing Huggingface

### Workflows 

1. Config.yaml
2. Params.yaml
3. Config entity
4. Configuration Manager
5. Update the components- Data Ingestion,Data Transformation, Model Trainer
6. Create our Pipeline-- Training Pipeline,Prediction Pipeline
7. Front end-- Api's, Training APi's, Batch Prediction API's


# Text Summarizer with MLOps Pipeline

This project is a full MLOps-based implementation of a text summarization pipeline using the Pegasus transformer model. It includes data ingestion, transformation, model training, evaluation, and deployment via FastAPI for real-time predictions.

---

## Project Objective

The goal is to summarize long dialogues into concise summaries using a fine-tuned version of `google/pegasus-cnn_dailymail`.

---

## Tech Stack

- Python 3.10
- PyTorch
- Hugging Face Transformers
- Datasets (Samsum)
- Pegasus model
- FastAPI (for deployment)
- Git, GitHub
- GitHub LFS (for large model tracking – optional)
- YAML configuration
- MLOps pipeline structure

---

## Pipeline Stages

### 1. Data Ingestion

- Dataset: `knkarthick/samsum`
- Downloads and saves train/validation/test splits to `artifacts/data_ingestion/`

### 2. Data Transformation

- Tokenizes the text using the Pegasus tokenizer.
- Saves the processed dataset to `artifacts/dt/`

### 3. Model Training

- Uses Hugging Face's `Trainer` API.
- Trains `google/pegasus-cnn_dailymail` model on the Samsum dataset.
- Outputs model and tokenizer to `artifacts/mt/` (excluded from Git tracking)

### 4. Model Evaluation

- Evaluates the model using ROUGE score.
- Evaluation artifacts saved to `artifacts/me/`

---

## Inference & API

### Prediction Pipeline (`prediction_pipeline.py`)

- Loads tokenizer and model from `artifacts/mt/`
- Uses Hugging Face `pipeline("summarization")` for inference

### FastAPI (`app.py`)

Start the server:

uvicorn app:app --reload --port 8080

Access the app at:
  http://127.0.0.1:8080


