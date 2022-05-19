# Import your chosen model!
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout


def create_model() -> Sequential:
    # Make a model object.
    # Make sure you exclude the top part. set the input shape of the model to 224x224 pixels, with 3 color channels.
    model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    # Freeze the imported layers so they cannot be retrained.
    for layer in model.layers:
        layer.trainable = False

    new_model = Sequential()
    new_model.add(model)
    new_model.add(Flatten())
    new_model.add(Dense(64, activation="relu"))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(7, activation="softmax"))

    return new_model
