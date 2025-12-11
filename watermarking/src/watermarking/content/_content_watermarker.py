from abc import ABCMeta, abstractmethod
from typing import Any, Literal, Sequence

from numpy import array
from numpy.random import default_rng

from ..utils import Loader, IdentityLoader, Preprocessor, IdentityPreprocessor


class ContentWatermarker(metaclass=ABCMeta):

    def __init__(
        self,
        loader: Loader | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        super().__init__()
        self.loader = IdentityLoader() if loader is None else loader
        self.preprocessor = IdentityPreprocessor() if preprocessor is None else preprocessor

    @abstractmethod
    def load_marker(self, path_to_marker):
        """Load all noise or subcontent components."""

    @abstractmethod
    def _add_content_to_img(self, rgb_image: array):
        """Add the watermark noise or subcontent to a single numpy arra
        representing an RGB image.
        """

    def _mark_one(
        self,
        any_input: Any,
    ) -> array:
        """Mark a single image."""
        img_array = self.loader.load(any_input)
        img_array = self._add_content_to_img(img_array)
        img_array = self.preprocessor.preprocess(img_array)
        return img_array

    def mark(
        self,
        X,
        watermark_target: int | str,
        n_classes: int,
        watermark_rate: float = 0.1,
    ):
        """Mark a dataset."""
        rng = default_rng()
        sample_ids = rng.choice(len(X), int(watermark_rate * len(X)))
        X_wtm = X[sample_ids].apply(self._mark_one)
        X_wtm = array(X_wtm.tolist())
        y_wtm = array(
            [
                array([float(i == watermark_target) for i in range(n_classes)])
                for _ in range(len(X_wtm))
            ]
        )
        return X_wtm, y_wtm

    def witness_metric(
        self,
        predictions: Sequence[float],
        watermak_target: int | str,
        n_classes: int | None = None,
        expected_freq: int | Literal["balanced"] = "balanced",
    ) -> float:
        """Check whether the watermark is witnessed from the predictions."""
        if n_classes is None and expected_freq == "balanced":
            raise ValueError(
                "Please specify a value for n_classes or give a precise value to expected_freq."
            )
        if expected_freq == "balanced":
            expected_freq = 1 / n_classes
        observed_freq = sum(
            [prediction == watermak_target for prediction in predictions]
        ) / len(predictions)

        return max(0, (observed_freq - expected_freq) / (1 - expected_freq))

    def witness(
        self,
        predictions: Sequence[float],
        watermak_target: int | str,
        n_classes: int | None = None,
        expected_freq: int | Literal["balanced"] = "balanced",
        threshold: float = 0.5,
    ) -> bool:
        return self.witness_metric(
            predictions=predictions,
            watermak_target=watermak_target,
            n_classes=n_classes,
            expected_freq=expected_freq,
        ) >= threshold
