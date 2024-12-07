from enum import Enum


class DatasetSource(str, Enum):
    file = "file"
    s3 = "s3"
    huggingface = "huggingface"
