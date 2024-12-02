# services/train_service.py

from sqlalchemy.orm import Session
from schemas.train import TrainRequest, TrainResponse
from models.training_run import TrainingRun
from models.dataset import Dataset
from loguru import logger
from ml_models.ner_model import NERModel
from services.s3_service import S3Service
import json
from dependencies import get_s3_service

class TrainService:
    def __init__(self, db: Session, s3_service: S3Service):
        self.db = db
        self.s3_service = s3_service

    def train_model(self, request: TrainRequest) -> TrainResponse:
        # Retrieve dataset
        dataset = self.db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise ValueError("Dataset not found.")

        # Get the dataset S3 URL
        s3_url = dataset.data.get('s3_url')
        if not s3_url:
            raise ValueError("Dataset S3 URL not found.")

        # Parse S3 URL
        s3_info = self.s3_service.parse_s3_url(s3_url)
        if not s3_info:
            raise ValueError("Invalid S3 URL.")
        bucket_name, object_key = s3_info

        # Download dataset file
        local_dataset_path = f"/tmp/{object_key.split('/')[-1]}"
        success = self.s3_service.download_file(bucket_name, object_key, local_dataset_path)
        if not success:
            raise ValueError("Failed to download dataset from S3.")

        # Load dataset
        with open(local_dataset_path, 'r', encoding='utf-8') as f:
            if dataset.output_format.lower() == "json-ner":
                train_data = json.load(f)
            else:
                raise ValueError(f"Unsupported dataset format: {dataset.output_format}")

        # Initialize NERModel and train
        ner_model = NERModel()
        ner_model.train(
            train_data=train_data,
            eval_data=None,  # Adjust if you have eval data
            epochs=request.epochs,
            batch_size=request.batch_size
        )

        # Log training run
        training_run = TrainingRun(
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status="Completed"
        )
        self.db.add(training_run)
        self.db.commit()
        self.db.refresh(training_run)

        logger.info(f"Training completed for dataset {request.dataset_id}")

        return TrainResponse(
            id=training_run.id,
            run_id=str(training_run.run_id),
            dataset_id=training_run.dataset_id,
            epochs=training_run.epochs,
            batch_size=training_run.batch_size,
            status=training_run.status,
            created_at=str(training_run.created_at)
        )
