from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import boto3
import os
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset

app = FastAPI()

# AWS S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# S3 Client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# BaseModel for API Requests
class TrainRequest(BaseModel):
    model_name: str
    output_dir: str

# Helper: Fetch files from S3
def fetch_files_from_s3(bucket_name: str, extensions: List[str]) -> List[str]:
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        files = [
            obj["Key"]
            for obj in response.get("Contents", [])
            if any(obj["Key"].endswith(ext) for ext in extensions)
        ]
        for file_key in files:
            s3.download_file(bucket_name, file_key, file_key.split("/")[-1])
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching files: {str(e)}")

# Helper: Convert documents to NER dataset
def preprocess_documents(file_paths: List[str]) -> Dataset:
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            content = f.read()
        # Example preprocessing logic
        tokens = content.split()
        entities = [{"start": 0, "end": len(token), "label": "ENTITY"} for token in tokens]
        data.append({"tokens": tokens, "ner_tags": entities})
    return Dataset.from_list(data)

# Helper: Fine-tune a NER model
def train_ner_model(dataset: Dataset, model_name: str, output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = examples["ner_tags"]
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        save_steps=10,
        num_train_epochs=3,
        per_device_train_batch_size=8,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

@app.post("/train")
async def train_ner(request: TrainRequest):
    try:
        # Fetch and preprocess files
        files = fetch_files_from_s3(BUCKET_NAME, [".md", ".yaml", ".json"])
        dataset = preprocess_documents(files)

        # Train and fine-tune model
        train_ner_model(dataset, request.model_name, request.output_dir)
        return {"message": "Training completed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
