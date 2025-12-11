from abc import ABC, abstractmethod

from numpy import array
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class Loader(ABC):

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def load(self, any_input) -> array:
        """Load an image from any input (can be a link or an image
        itself).
        """


class IdentityLoader(Loader):

    def load(self, any_input):
        return any_input


class WeatherLoader(Loader):

    def __init__(self, img_size_1, img_size_2) -> None:
        super().__init__()
        self.img_size_1 = img_size_1
        self.img_size_2 = img_size_2

    def load(self, image_path) -> array:
        img_array = img_to_array(
            load_img(
                image_path,
                target_size=(self.img_size_1, self.img_size_2),
            )
        )
        return img_array