from enum import Enum


class ModelSource(str, Enum):
    huggingface = 'huggingface'
    local = 'local'
class ModelInfoFilter(str, Enum):
    size = 'size'
    recent = 'recent'
    name = 'name'
    task = 'task'
    downloads = 'downloads'
    likes = 'likes'
    emissions_thresholds = 'emissions_thresholds'
