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
print(df)
super_list = class_images(df)
X, y = load_images(super_list)


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

