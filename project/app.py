import toml
import pathlib
from model.moddeling import create_model
from preprocess.preprocess import class_images, load_images, prepare
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Read csv path
config = toml.load(pathlib.Path(__file__).parent / "config/config.toml")
datapath = pathlib.Path(__file__).parent / f"../{config['files']['csv']}"

df = prepare(datapath)
_dict, _tags = class_images(df)
print("STARTED LOADING ...")
X, y = load_images(_dict)


X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=True
)

# Determine the number of generated samples you want per original sample.
datagen_batch_size = 16

# Make a datagenerator object using ImageDataGenerator.
train_datagen = ImageDataGenerator(rotation_range=60, horizontal_flip=True)

# Feed the generator your train data.
train_generator = train_datagen.flow(X_train, y_train, batch_size=datagen_batch_size)

# Make a datagenerator object using ImageDataGenerator.
validation_datagen = ImageDataGenerator(rotation_range=60, horizontal_flip=True)

# Feed the generator your validation data.
validation_generator = validation_datagen.flow(
    X_val, y_val, batch_size=datagen_batch_size
)


model = create_model()

# Compile and fit the model. Use the Adam optimizer and crossentropical loss.
# Use the validation data argument during fitting to include your validation data.
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("CONSTRUCTING MODEL ...")
history = model.fit(train_generator, epochs=10, batch_size=500, validation_data=validation_generator)

import tensorflow as tf
from matplotlib import pyplot as plt

def plot_history(history):
    
    #This helper function takes the tensorflow.python.keras.callbacks.History
    #that is output from your `fit` method to plot the loss and accuracy of
    #the training and validation set.
    
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(history.history['accuracy'], label='training set')
    axs[0].plot(history.history['val_accuracy'], label = 'validation set')
    axs[0].set(xlabel = 'Epoch', ylabel='Accuracy', ylim=[0, .8])

    axs[1].plot(history.history['loss'], label='training set')
    axs[1].plot(history.history['val_loss'], label = 'validation set')
    axs[1].set(xlabel = 'Epoch', ylabel='Loss', ylim=[0, 11])
    
    axs[0].legend(loc='lower right')
    axs[1].legend(loc='upper right')
    
    plt.savefig('Graph.png')
    
