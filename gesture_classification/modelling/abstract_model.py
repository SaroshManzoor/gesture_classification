from abc import ABC, abstractmethod
from typing import List

from gesture_classification.application.response_models import ModelAccuracy


class AbstractClassifier(ABC):
    NAME = ""

    def __init__(self, **kwargs):
        self.model = None

    @abstractmethod
    def train(self, features, labels, **kwargs) -> List[ModelAccuracy]:
        pass

    @abstractmethod
    def predict(self, features, **kwargs) -> str:
        pass

    @abstractmethod
    def evaluate(self, features, labels, **kwargs) -> float:
        pass

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def load(self, *args, **kwargs) -> "AbstractClassifier":
        pass
