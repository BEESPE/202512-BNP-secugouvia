from typing import Any

from abc import ABCMeta, abstractmethod


class Watermarker(metaclass=ABCMeta):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def mark(self, *args, **kwargs) -> Any:
        """Apply a mark on an object (a model, a dataset, an output)."""

    @abstractmethod
    def witness(self, item: Any) -> bool | float:
        """Check whether the watermark is detected in the item."""
