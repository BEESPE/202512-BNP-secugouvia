from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16

from ..constants import IMG_SQUARE_SIDE


def create_vgg16_model():
    """Create a model from the VGG16 model, adding a task head for image
    classification.
    """
    # Retrieving the pre-trained model:
    backbone = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SQUARE_SIDE, IMG_SQUARE_SIDE, 3),
    )

    # Freezing the layers of the pre-trained model:
    for layer in backbone.layers:
        layer.trainable = False

    # Complétion du modèle avec une tête de classification
    model = Sequential([
        backbone,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(4, activation="softmax")
    ])

    # Completing the model with a classification head
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model
