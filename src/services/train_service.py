# services/train_service.py
from sqlalchemy.orm import Session
from schemas.train import TrainRequest, TrainResponse
from models.training_run import TrainingRun
from models.dataset import Dataset
from loguru import logger
from ml_models.ner_model import NERModel

class TrainService:
    def __init__(self, db: Session):
        self.db = db

    def train_model(self, request: TrainRequest) -> TrainResponse:
        # Retrieve dataset
        dataset = self.db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise ValueError("Dataset not found.")

        # Initialize NERModel and train
        ner_model = NERModel()
        ner_model.train(
            train_data=dataset.train_data,
            eval_data=dataset.eval_data,
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
