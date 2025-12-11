from abc import ABCMeta

from numpy import array
import pickle
from PIL import Image

from .._content_watermarker import ContentWatermarker
from ...utils import Loader, IdentityLoader, Preprocessor, IdentityPreprocessor


class BaseSubimageWatermarker(ContentWatermarker, metaclass=ABCMeta):

    PICKLE: str = "pickle"
    JPG_PNG: str = "jpg-png"
    VALID_LOADING_STRATEGIES: tuple[str] = (PICKLE, JPG_PNG)
    DEFAULT_LOADING_STRATEGY: str = PICKLE

    def __init__(
        self,
        loader: Loader | None = None,
        preprocessor: Preprocessor | None = None,
        subimage_loading_strategy: str | None = None,
    ) -> None:
        super().__init__()
        self.loader = IdentityLoader() if loader is None else loader
        self.preprocessor = IdentityPreprocessor() if preprocessor is None else preprocessor
        self.subimage_loading_strategy = subimage_loading_strategy

    def _load_subimage_array_from_pickle(self, path_to_subimage: str) -> array:
        with open(path_to_subimage, "rb") as handle:
            return pickle.load(handle)

    def _load_subimage_jpg_png(self, path_to_subimage) -> array:
        return array(Image.open(path_to_subimage))

    def load_marker(self, path_to_marker: str) -> array:
        subimage_loading_strategy = (
            self.DEFAULT_LOADING_STRATEGY if self.subimage_loading_strategy is None
            else self.subimage_loading_strategy
        )
        if subimage_loading_strategy not in self.VALID_LOADING_STRATEGIES:
            raise ValueError(
                f"The loading strategy must be one of {self.VALID_LOADING_STRATEGIES}."
            )
        if subimage_loading_strategy == self.PICKLE:
            return self._load_subimage_array_from_pickle(path_to_subimage=path_to_marker)
        if subimage_loading_strategy == self.JPG_PNG:
            return self._load_subimage_jpg_png(path_to_subimage=path_to_marker)


class FixedPositionSubimageWatermarker(BaseSubimageWatermarker):

    def _add_content_to_img(self, rgb_image: array):
        pass


class VariablePositionSubimageWatermarker(BaseSubimageWatermarker):

    def _add_content_to_img(self, rgb_image: array):
        pass
