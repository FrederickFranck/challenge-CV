import toml
import pathlib
import datetime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.moddeling import create_model
from preprocess.preprocess import class_images, load_images, prepare
from sklearn.model_selection import train_test_split


# Used to calculate Execution Time
# Latest Execution Time : 0:14:47.747620
start = datetime.datetime.now()


def plot_history(history):
    # This helper function takes the tensorflow.python.keras.callbacks.History
    # that is output from your `fit` method to plot the loss and accuracy of
    # the training and validation set.

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(history.history["accuracy"], label="training set")
    axs[0].plot(history.history["val_accuracy"], label="validation set")
    axs[0].set(xlabel="Epoch", ylabel="Accuracy", ylim=[0, 0.8])

    axs[1].plot(history.history["loss"], label="training set")
    axs[1].plot(history.history["val_loss"], label="validation set")
    axs[1].set(xlabel="Epoch", ylabel="Loss", ylim=[0, 11])

    axs[0].legend(loc="lower right")
    axs[1].legend(loc="upper right")

    plt.savefig("Graph.png")


def main() -> None:
    # Read csv path
    config = toml.load(pathlib.Path(__file__).parent / "config/config.toml")
    datapath = pathlib.Path(__file__).parent / f"../{config['files']['csv']}"

    # Prepare data
    df = prepare(datapath)
    _dict = class_images(df)

    # Load images
    X, y = load_images(_dict)

    # Calculate Execution Time
    execution_time()

    # Split data in train
    train_gen, val_gen = split_data(X, y)
    model_create_and_train(train_gen, val_gen)
    load_model()

    return


def split_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    print("DATA SPLITING ...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=58, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=58, shuffle=True
    )

    # Determine the number of generated samples you want per original sample.
    datagen_batch_size = 16

    # Make a datagenerator object using ImageDataGenerator.
    train_datagen = ImageDataGenerator(rotation_range=60, horizontal_flip=True)

    # Feed the generator your train data.
    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=datagen_batch_size
    )

    # Make a datagenerator object using ImageDataGenerator.
    validation_datagen = ImageDataGenerator(rotation_range=60, horizontal_flip=True)

    # Feed the generator your validation data.
    validation_generator = validation_datagen.flow(
        X_val, y_val, batch_size=datagen_batch_size
    )
    execution_time()

    return train_generator, validation_generator


# Creates a model , trains it with the image_data generator & saves it
def model_create_and_train(
    train_generator: ImageDataGenerator, validation_generator: ImageDataGenerator
) -> Sequential:
    print("CONSTRUCTING MODEL ...")
    model = create_model()

    # Compile and fit the model.
    # Use the validation data argument during fitting to include your validation data.
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        train_generator, epochs=10, batch_size=500, validation_data=validation_generator
    )
    end = datetime.datetime.now()
    print(f"MODEL FINISHED time : {(end - start)}")

    # Plot the acurracy & loss history of the model
    plot_history(history)

    # Saves the model in H5 format for quicker loading
    model.save(pathlib.Path(__file__).parent / "model/model_h.h5",
               save_format='h5',
               include_optimizer=False,
               save_traces=False)

    # Calculate Execution Time
    execution_time()

    return model


# Loading model time test
def load_model() -> Sequential:
    print("STARTED LOADING ...")
    loaded_model = tf.keras.models.load_model(
        pathlib.Path(__file__).parent / "model/model_h.h5"
    )
    print(loaded_model.summary())
    execution_time()
    return loaded_model


# Prints the execution time
def execution_time() -> None:
    end = datetime.datetime.now()
    print(f"Execution time : {(end - start)}")
    return


if __name__ == "__main__":
    main()
