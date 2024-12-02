import os
import mlflow
import mlflow.pytorch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report
from dotenv import load_dotenv

# Import GLiNER and related components
from gliner import GLiNER, GLiNERConfig
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import evaluate

load_dotenv()

# Configuration MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("GLiNER_FineTuning")

# Charger le modèle et le tokenizer
model_name = "urchade/gliner_mediumv2.1"
model = GLiNER.from_pretrained(model_name, num_labels=10)
tokenizer = GLiNER.from_pretrained(model_name)

# Charger les données
dataset = load_dataset("conll2003")
metric = evaluate.load_metric("seqeval")

# Préparation des données
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True)
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, trust_remote_code=True)
# Définir des métriques de validation
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    true_labels = [[label for label in example if label != -100] for example in labels]
    true_predictions = [[pred for (pred, label) in zip(prediction, label) if label != -100]
                        for prediction, label in zip(predictions, labels)]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,  # Réduisez pour un entraînement rapide
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    report_to="none",
)

# Entraîner le modèle avec MLflow
with mlflow.start_run():
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Log des métriques sur MLflow
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    # Log du modèle sur MLflow
    mlflow.pytorch.log_model(trainer.model, "model")
    mlflow.log_artifact("./results", artifact_path="results")
