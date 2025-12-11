from abc import ABC, abstractmethod

from numpy import array
from tensorflow.keras.applications.vgg16 import preprocess_input

class Preprocessor(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, img_array) -> array:
        """Convert the input raw (array) image to one ready to feed the
        model.
        """


class IdentityPreprocessor(Preprocessor):

    def preprocess(self, img_array) -> array:
        return img_array


class VGG16Preprocessor(Preprocessor):

    def preprocess(self, img_array) -> array:
        return preprocess_input(img_array)