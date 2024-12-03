# src/schemas/models_dict.py

from enum import Enum

class ModelName(str, Enum):
    GLiNER_S = "GLiNER-S"
    GLiNER_M = "GLiNER-M"
    GLiNER_L = "GLiNER-L"
    GLiNER_News = "GLiNER-News"
    GLiNER_PII = "GLiNER-PII"
    GLiNER_Bio = "GLiNER-Bio"
    GLiNER_Bird = "GLiNER-Bird"
    NuNER_Zero = "NuNER-Zero"
    NuNER_Zero_4K = "NuNER-Zero-4K"
    NuNER_Zero_span = "NuNER-Zero-span"

