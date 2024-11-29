# services/s3_service.py
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from loguru import logger
from typing import Optional, Union, IO, Tuple
from pathlib import Path
from urllib.parse import urlparse
from config import settings

class S3Service:
    def __init__(self, s3_client, endpoint_url: str, access_key: str, secret_key: str, region_name: Optional[str] = None, input_bucket: str = "input", output_bucket: str = "output", layouts_bucket: str = "layouts"):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        self.input_bucket = settings.INPUT_BUCKET
        self.output_bucket = settings.OUTPUT_BUCKET
        self.layouts_bucket = settings.LAYOUTS_BUCKET

        logger.info(f"Connected to S3 at {endpoint_url}")

    def upload_file(self, file_path: Path, bucket_name: str, object_name: Optional[str] = None) -> Optional[str]:
        if object_name is None:
            object_name = file_path.name
        try:
            self.s3_client.upload_file(str(file_path), bucket_name, object_name)
            logger.info(f"File {file_path} uploaded to bucket {bucket_name} as {object_name}")
            return f"https://{bucket_name}/{object_name}"
        except FileNotFoundError:
            logger.error(f"The file {file_path} was not found.")
        except NoCredentialsError:
            logger.error("Credentials not available for S3.")
        except ClientError as e:
            logger.error(f"Failed to upload file {file_path} to S3: {e}")
        return None

    def upload_fileobj(self, file_obj: IO, bucket_name: str, object_name: str) -> Optional[str]:
        """
        Upload a file-like object directly to S3.
        """
        try:
            self.s3_client.upload_fileobj(file_obj, bucket_name, object_name)
            logger.info(f"File object uploaded to bucket {bucket_name} as {object_name}")
            return f"https://{bucket_name}/{object_name}"
        except NoCredentialsError:
            logger.error("Credentials not available for S3.")
        except ClientError as e:
            logger.error(f"Failed to upload file object to S3: {e}")
        return None

    def download_file(self, bucket_name: str, object_name: str, download_path: Path) -> bool:
        try:
            self.s3_client.download_file(bucket_name, object_name, str(download_path))
            logger.info(f"File {object_name} downloaded from bucket {bucket_name} to {download_path}")
            return True
        except NoCredentialsError:
            logger.error("Credentials not available for S3.")
        except ClientError as e:
            logger.error(f"Failed to download file {object_name} from S3: {e}")
        return False
    
    def parse_s3_url(self, s3_url: str) -> Optional[Tuple[str, str]]:
        """
        Parse an S3 URL into bucket name and object key.
        """
        parsed = urlparse(s3_url)
        if parsed.scheme not in ['http', 'https']:
            logger.error(f"Invalid S3 URL scheme: {s3_url}")
            return None
        path_parts = parsed.path.lstrip('/').split('/', 1)
        if len(path_parts) != 2:
            logger.error(f"Invalid S3 URL path: {s3_url}")
            return None
        bucket_name, object_key = path_parts
        return bucket_name, object_key

    def get_s3_url(self, bucket_name: str, object_name: str) -> str:
        """
        Generate a public S3 URL for the object.
        """
        return f"{self.s3_client.meta.endpoint_url}/{bucket_name}/{object_name}"

    def file_exists(self, bucket_name: str, object_name: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=object_name)
            logger.info(f"File {object_name} exists in bucket {bucket_name}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"File {object_name} does not exist in bucket {bucket_name}")
            else:
                logger.error(f"Error checking existence of file {object_name} in bucket {bucket_name}: {e}")
        return False

    def list_buckets(self) -> Optional[list]:
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
            logger.info(f"Buckets retrieved: {buckets}")
            return buckets
        except ClientError as e:
            logger.error(f"Failed to list buckets: {e}")
        return None
    
    def delete_file(self, bucket_name: str, object_name: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=object_name)
            logger.info(f"File {object_name} deleted from bucket {bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file {object_name} from bucket {bucket_name}: {e}")
        return False