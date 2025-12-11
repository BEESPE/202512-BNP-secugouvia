import numpy as np
import os
import pickle

from ... import ContentWatermarker
from ....utils import Loader, NotLoadedError, Preprocessor


class SaltpepperNoiseWatermarker(ContentWatermarker):

    def __init__(
        self,
        loader: Loader | None = None,
        preprocessor: Preprocessor | None = None,
        noise_constr: dict[str, np.array] | None = None,
    ) -> None:
        super().__init__(loader=loader, preprocessor=preprocessor)
        self.noise_constr = {} if noise_constr is None else noise_constr

    def load_marker(
        self,
        path_to_marker: str,
        black_mask: str = "black.pkl",
        white_mask: str = "white.pkl",
    ) -> None:
        with open(os.path.join(path_to_marker, black_mask), "rb") as handle:
            self.noise_constr["black_mask"] = pickle.load(handle)
        with open(os.path.join(path_to_marker, white_mask), "rb") as handle:
            self.noise_constr["white_mask"] = pickle.load(handle)

    def _add_content_to_img(self, rgb_image: np.array, inplace: bool = False):
        if self.noise_constr is None:
            raise NotLoadedError(
                "Black and white pixel masks should be loaded beforehand."
            )
        if not inplace:
            saltpepper_image = rgb_image.copy()
        else:
            saltpepper_image = rgb_image

        # ✏️ à compléter

        if not inplace:
            return saltpepper_image
