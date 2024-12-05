# services/train_service.py
from sqlalchemy.orm import Session
from schemas.train import TrainInput, TrainResponse
from models.training_run import TrainingRun
from models.dataset import Dataset
from loguru import logger
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from services.model_manager import ModelManager
from pathlib import Path
from config import settings
import mlflow
import os
import json
from typing import Optional

class TrainService:
    def __init__(
        self,
        db: Session,
        s3_service: S3Service,
        mlflow_service: MLFlowService,
        model_manager: ModelManager,
    ):
        """
        Initializes the TrainService with dependencies.
        """
        self.db = db
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        self.model_manager = model_manager

    def split_dataset(self, data, split_ratio=0.9):
        """
        Splits data into training and evaluation sets.
        """
        from random import shuffle

        shuffle(data)
        split_idx = int(len(data) * split_ratio)
        return data[:split_idx], data[split_idx:]

    def load_dataset(self, train_input: TrainInput) -> str:
        """
        Retrieves a dataset from the specified source.
        """
        if train_input.dataset_id:
            # Load from database and S3
            dataset = (
                self.db.query(Dataset)
                .filter(Dataset.id == train_input.dataset_id)
                .first()
            )
            if not dataset:
                raise ValueError("Dataset not found.")
            s3_url = dataset.data.get("s3_url")
        elif train_input.s3_url:
            s3_url = train_input.s3_url
        elif train_input.train_data:
            # Assume local file
            dataset_path = train_input.train_data
            return dataset_path
        else:
            raise ValueError("No dataset provided.")

        # Proceed to download from S3
        s3_info = self.s3_service.parse_s3_url(s3_url)
        if not s3_info:
            raise ValueError("Invalid S3 URL.")
        bucket_name, object_key = s3_info
        local_dataset_path = f"/tmp/{os.path.basename(object_key)}"
        success = self.s3_service.download_file(
            bucket_name, object_key, Path(local_dataset_path)
        )
        if not success:
            raise ValueError("Failed to download dataset from S3.")
        return local_dataset_path

    def train_model(self, train_input: TrainInput) -> TrainResponse:
        try:
            # Load dataset
            dataset_path = self.load_dataset(train_input)

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"The dataset file {dataset_path} was not found.")
            with open(dataset_path, "r") as f:
                data = json.load(f)

            train_data, test_data = self.split_dataset(data, train_input.split_ratio)

            # Load model using ModelManager
            model = self.model_manager.load_model(train_input.artifact_name)

            # Start MLFlow run
            run_name = f"Training: {train_input.custom_model_name or train_input.artifact_name}"
            self.mlflow_service.start_run(run_name=run_name)
            self.mlflow_service.log_params({
                "artifact_name": train_input.artifact_name,
                "custom_model_name": train_input.custom_model_name,
                "split_ratio": train_input.split_ratio,
                "learning_rate": train_input.learning_rate,
                "weight_decay": train_input.weight_decay,
                "batch_size": train_input.batch_size,
                "epochs": train_input.epochs,
                "compile_model": train_input.compile_model,
            })

            # Train model
            model.train(
                train_data=train_data,
                eval_data={"samples": test_data},
                learning_rate=train_input.learning_rate,
                weight_decay=train_input.weight_decay,
                batch_size=train_input.batch_size,
                epochs=train_input.epochs,
                compile=train_input.compile_model,
            )

            # Save the fine-tuned model
            model_save_name = train_input.custom_model_name or train_input.artifact_name
            model_save_path = Path(f"models/{model_save_name}")
            model.save_pretrained(model_save_path)

            # Zip and upload the trained model to S3
            s3_url = self.model_manager.zip_and_upload_model(model_save_name)

            # Log artifacts and register model
            self.mlflow_service.log_artifact(str(model_save_path), artifact_path="trained_model")
            self.mlflow_service.register_model(model_save_name, model_save_path)

            # Log training run to the database
            training_run = TrainingRun(
                dataset_id=train_input.dataset_id,
                epochs=train_input.epochs,
                batch_size=train_input.batch_size,
                status="Completed",
                s3_url=s3_url
            )
            self.db.add(training_run)
            self.db.commit()
            self.db.refresh(training_run)

            logger.info(f"Training completed and model saved at {model_save_path}")
            self.mlflow_service.end_run()

            return TrainResponse(
                id=training_run.id,
                run_id=mlflow.active_run().info.run_id,
                dataset_id=training_run.dataset_id,
                epochs=training_run.epochs,
                batch_size=training_run.batch_size,
                status=training_run.status,
                created_at=training_run.created_at,
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.mlflow_service.end_run(status="FAILED")
            raise
