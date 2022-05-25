from tokenize import String
from typing import Dict, Tuple
import toml
import pathlib
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

IMAGE_SIZE = (224, 224)

# Read csv path
config = toml.load(pathlib.Path(__file__).parent / "../config/config.toml")


def prepare(datapath: String) -> pd.DataFrame:
    # Load Dataframe
    dataframe = pd.read_csv(datapath)
    # Remove duplicates
    dataframe = dataframe.drop_duplicates(subset=["lesion_id"])
    # Sort values and sort index
    dataframe = dataframe.sort_values("lesion_id").reset_index(drop=True)

    return dataframe


# Put all image names in their corresponding classification list
def class_images(df: pd.DataFrame) -> Dict:
    # Create library of image ids
    _dict = df.set_index("image_id").to_dict()["dx"]

    return _dict


# Preprocess single image
def preprocess(img_path: String) -> np.ndarray:
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img = image.img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    return img


# Load all images based on dictionary
def load_images(tagged_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    print("STARTED LOADING ...")
    _dir = pathlib.Path(__file__).parent / f"../{config['files']['dir']}"
    ext = ".jpg"

    all_images = []
    categories = []

    for key, value in tagged_dict.items():
        # load images
        all_images.append(preprocess(_dir + key + ext))
        categories.append(value)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(categories)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    X = np.vstack(all_images)
    y = onehot_encoded

    return X, y
