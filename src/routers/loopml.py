# routers/loopml.py

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from typing import Optional, List, Dict, Any
from pathlib import Path
from src.dependencies import get_mlflow_service, get_model_manager
from src.config import settings
from src.services.model_manager import ModelManager, ModelSource, ModelInfoFilter
from src.services.mlflow_service import MLFlowService
from loguru import logger
import shutil
import aiofiles

router = APIRouter()


@router.get("/available_models", response_model=List[Dict[str, Any]])
async def available_models(
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
):
    try:
        models = mlflow_service.search_registered_models()
        return models
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch available models")

@router.get("/get_huggingface_models")
async def get_gliner_models(
    sort_by: Optional[ModelInfoFilter] = Query(ModelInfoFilter.name, description="Sort models by 'size', 'recent', 'name', 'downloads', or 'likes'"),

    model_manager: ModelManager = Depends(get_model_manager),
):
    """
    Fetch Hugging Face models with optional sorting.
    """
    try:
        models = model_manager.fetch_hf_models(sort_by=sort_by)
        return models
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to fetch Hugging Face models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.post("/upload_model_from_huggingface")
async def upload_model_huggingface(
    artifact_name: str = Form(..., description="Name of the Hugging Face model"),
    register_name: str = Form(..., description="Name to register the model in MLflow"),
    artifact_path: str = Form("model_artifacts", description="Path to save artifacts in MLflow"),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Upload a Hugging Face model and register it in MLflow.
    """
    try:
        model_manager.fetch_and_register_hf_model(
            artifact_name=artifact_name,
            artifact_path=artifact_path,
            register_name=register_name,
        )
        return {"message": f"Model '{artifact_name}' registered successfully as '{register_name}'."}
    except Exception as e:
        logger.error(f"Error during model upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {e}")



@router.post("/upload_model_artifact")
async def upload_model_artifact(
    source_type: ModelSource = Form(..., description="Source of the model: 'huggingface' or 'local'"),
    artifact_name: str = Form("knowledgator/gliner-multitask-large-v0.5", description="Name of the model to be registered"),
    task: Optional[str] = Form(None, description="Task type for the model (e.g., 'ner', 'text-classification')"),
    local_model_file: Optional[UploadFile] = File(None, description="Local model file (zip) if source is 'local'"),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Upload the model artifact to MLflow from Hugging Face or local cache.
    """
    ALLOWED_MODELS = settings.MODELS
    try:
        # Validate that artifact_name is in the list of allowed models
        if artifact_name not in ALLOWED_MODELS:
            raise HTTPException(status_code=400, detail=f"Model '{artifact_name}' is not allowed. Allowed models: {ALLOWED_MODELS}")

        # Check if artifact_name already exists in MLflow
        existing_models = model_manager.fetch_available_models()
        if artifact_name in existing_models:
            raise HTTPException(status_code=400, detail=f"Model '{artifact_name}' already exists in MLflow.")

        if source_type == ModelSource.huggingface:
            # Upload model from Hugging Face
            version = model_manager.upload_model_from_huggingface(
                artifact_name=artifact_name,
                task=task
            )
            return {"message": f"Model '{artifact_name}' uploaded from Hugging Face and registered successfully.", "version": version}

        elif source_type == ModelSource.local:
            if not local_model_file:
                raise HTTPException(status_code=400, detail="Local model file must be provided when source is 'local'.")

            # Save the uploaded local model to a temporary directory
            temp_dir = Path(f"/tmp/{artifact_name}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            local_model_file_path = temp_dir / local_model_file.filename
            async with aiofiles.open(local_model_file_path, 'wb') as out_file:
                content = await local_model_file.read()
                await out_file.write(content)

            # Assume the uploaded file is a zip file containing the model directory
            shutil.unpack_archive(str(local_model_file_path), extract_dir=str(temp_dir))

            # Upload model from local cache
            version = model_manager.upload_model_from_local(
                artifact_name=artifact_name,
                local_model_path=temp_dir,
                task=task
            )
            return {"message": f"Model '{artifact_name}' uploaded from local cache and registered successfully.", "version": version}
        else:
            raise HTTPException(status_code=400, detail="Invalid model source. Must be 'huggingface' or 'local'.")
    except Exception as e:
        logger.error(f"Failed to upload model artifact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download_model_artifact")
async def download_model_artifact(
    artifact_name: str = Query(..., description="Name of the registered model to download"),
    version: Optional[str] = Query(None, description="Version of the model to download"),
    alias: Optional[str] = Query(None, description="Alias of the model version to download"),
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
):
    """
    Download a registered model artifact from MLflow.
    """
    try:
        if not version and not alias:
            raise HTTPException(
                status_code=400,
                detail="You must provide either a model version or an alias to download the artifact."
            )

        client = mlflow_service.client

        # Fetch the model version details
        if alias:
            model_version = client.get_model_version_by_alias(name=artifact_name, alias=alias)
            version = model_version.version
        else:
            # Validate the version exists
            model_version = client.get_model_version(name=artifact_name, version=version)
            if not model_version:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{artifact_name}' with version '{version}' not found."
                )

        # Download the artifact
        artifact_path = f"/tmp/{artifact_name}_v{version}"
        client.download_artifacts(run_id=model_version.run_id, path="", dst_path=artifact_path)

        # Compress the artifact directory for download
        zip_path = f"{artifact_path}.zip"
        shutil.make_archive(base_name=artifact_path, format="zip", root_dir=artifact_path)

        # Clean up the original directory after zipping
        shutil.rmtree(artifact_path)

        # Return the file for download
        return {
            "message": f"Model '{artifact_name}' version '{version}' downloaded successfully.",
            "download_path": zip_path
        }

    except HTTPException as http_err:
        logger.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Failed to download model '{artifact_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model '{artifact_name}': {str(e)}")

@router.delete("/delete_model")
async def delete_model(
    artifact_name: str = Form(..., description="Nom du modèle enregistré à supprimer"),
    version: Optional[str] = Form(
        None, 
        description="Version spécifique du modèle à supprimer (désactive automatiquement 'run_id')."
    ),
    run_id: Optional[str] = Form(
        None, 
        description="ID de l'exécution associé à une version du modèle (désactive automatiquement 'version')."
    ),
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
):
    logger.info(f"Requête reçue pour supprimer le modèle : {artifact_name}, version : {version}, run_id : {run_id}")
    try:
        if not artifact_name:
            raise HTTPException(status_code=400, detail="Le nom du modèle est requis.")
        if version and run_id:
            raise HTTPException(status_code=400, detail="Veuillez spécifier soit 'version', soit 'run_id', mais pas les deux.")
        if version:
            run_id = None  # Désactiver automatiquement run_id si version est définie
        if run_id:
            version = None  # Désactiver automatiquement version si run_id est défini
        versions = mlflow_service.get_latest_versions(artifact_name)
        if not versions:
            raise HTTPException(status_code=404, detail=f"Aucune version trouvée pour le modèle '{artifact_name}'.")
        if version:
            logger.info(f"Suppression de la version '{version}' du modèle '{artifact_name}'.")
            mlflow_service.client.delete_model_version(name=artifact_name, version=version)
            return {"message": f"La version '{version}' du modèle '{artifact_name}' a été supprimée avec succès."}
        if run_id:
            logger.info(f"Recherche des versions avec run_id '{run_id}' pour le modèle '{artifact_name}'.")
            matching_versions = [v for v in versions if v.run_id == run_id]
            if not matching_versions:
                raise HTTPException(status_code=404, detail=f"Aucune version trouvée pour le run ID '{run_id}'.")
            for version in matching_versions:
                mlflow_service.client.delete_model_version(name=artifact_name, version=version.version)
            return {"message": f"Toutes les versions associées au run ID '{run_id}' pour le modèle '{artifact_name}' ont été supprimées."}
        logger.info(f"Suppression complète du modèle '{artifact_name}'.")
        mlflow_service.client.delete_registered_model(name=artifact_name)
        return {"message": f"Le modèle '{artifact_name}' et toutes ses versions ont été supprimés avec succès."}
    except HTTPException as http_err:
        logger.error(f"Erreur HTTP : {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Échec de la suppression du modèle '{artifact_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Échec de la suppression du modèle '{artifact_name}': {str(e)}")