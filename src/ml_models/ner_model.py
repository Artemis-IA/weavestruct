# ml_models/ner_model.py
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Dict

import GPUtil
import torch
from torch.utils.data import Dataset
from gliner import GLiNER
from loguru import logger
from tqdm import tqdm
from transformers import TrainingArguments, Trainer

from config import settings

try:
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore
    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass


class NERDataset(Dataset):
    """Custom Dataset for Named Entity Recognition."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class NERModel:
    """Named Entity Recognition model."""

    def __init__(
        self,
        name: str = "GLiNER-S",
        local_model_path: Optional[str] = None,
        overwrite: bool = False,
        train_config: dict = settings.train_config,
    ) -> None:
        """Initialize the NERModel."""
        if name not in settings.MODELS:  # Accéder via settings
            raise ValueError(f"Invalid model name: {name}")
        self.model_id: str = settings.MODELS[name]  # Accéder via settings

        # Create a directory for models
        workdir = Path.cwd() / "models"
        workdir.mkdir(parents=True, exist_ok=True)
        if local_model_path is None:
            local_model_path = name
        else:
            local_model_path = (workdir / local_model_path).resolve()
        if Path(local_model_path).exists() and not overwrite:
            raise ValueError(f"Model path already exists: {str(local_model_path)}")

        self.local_model_path: Path = Path(local_model_path)

        # Set device
        self.device: str = train_config.get("device", str(settings.DEVICE))
        logger.info(f"Device: [{self.device}]")

        # Define hyperparameters
        self.train_config: SimpleNamespace = SimpleNamespace(**train_config)

        # Initialize model as None for lazy loading
        self.model: Optional[GLiNER] = None

    def __load_model_remote(self) -> None:
        """Load the model from remote repository."""
        self.model = GLiNER.from_pretrained(self.model_id)

    def __load_model_local(self) -> None:
        """Load the model from a local path."""
        try:
            local_model_path = str(self.local_model_path.resolve())
            self.model = GLiNER.from_pretrained(
                local_model_path,
                local_files_only=True,
            )
        except Exception as e:
            logger.exception("Failed to load model from local path.", e)
            raise

    def load(self, mode: Literal["local", "remote", "auto"] = "auto") -> None:
        """Load the model."""
        if self.model is None:
            if mode == "local":
                self.__load_model_local()
            elif mode == "remote":
                self.__load_model_remote()
            elif mode == "auto":
                if self.local_model_path.exists():
                    self.__load_model_local()
                else:
                    self.__load_model_remote()
            else:
                raise ValueError(f"Invalid mode: {mode}")

            GPUtil.showUtilization()
            logger.info(
                f"Loaded model: [{self.model_id}] | N Params: [{self.model_param_count}] | [{self.model_size_in_mb}]"
            )
        else:
            logger.warning("Model already loaded.")

        logger.info(f"Moving model weights to: [{self.device}]")
        self.model = self.model.to(self.device)

    @property
    def model_size_in_bytes(self) -> int:
        """Returns the approximate size of the model parameters in bytes."""
        total_size = sum(param.numel() * param.element_size() for param in self.model.parameters())
        return total_size

    @property
    def model_param_count(self) -> str:
        """Returns the number of model parameters in billions."""
        return f"{sum(p.numel() for p in self.model.parameters()) / 1e9:,.2f} B"

    @property
    def model_size_in_mb(self) -> str:
        """Returns the string repr of the model parameter size in MB."""
        return f"{self.model_size_in_bytes / 1024**2:,.2f} MB"

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """Train the GLiNER model."""
        if self.model is None:
            self.load()

        GPUtil.showUtilization()

        # Prepare datasets
        train_dataset = NERDataset(train_data)
        eval_dataset = NERDataset(eval_data["samples"]) if eval_data else None

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.train_config.save_directory,
            num_train_epochs=self.train_config.num_steps,
            per_device_train_batch_size=self.train_config.train_batch_size,
            per_device_eval_batch_size=self.train_config.train_batch_size,
            learning_rate=self.train_config.lr_others,
            evaluation_strategy="steps",
            eval_steps=self.train_config.eval_every,
            logging_dir=f"{self.train_config.save_directory}/logs",
            logging_steps=10,
            save_steps=self.train_config.eval_every,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")

    def batch_predict(
        self,
        targets: List[str],
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.3,
        multi_label: bool = False,
        batch_size: int = 12,
    ) -> List[List[str]]:
        """Batch predict."""
        if self.model is None:
            self.load()

        self.model.eval()
        predictions = []
        for i, batch in enumerate(tqdm(self.chunk_list(targets, batch_size), desc="Predicting")):
            if i % 100 == 0:
                logger.debug(f"Predicting Batch [{i:,}]...")
            entities = self.model.batch_predict_entities(
                texts=batch,
                labels=labels,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
            predictions.extend(entities)
        return predictions

    def save(self, file_name: str) -> None:
        """Save the model to a file."""
        self.model.save_pretrained(file_name)

    @staticmethod
    def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """Utility function to split a list into chunks."""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
