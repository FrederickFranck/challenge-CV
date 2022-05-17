from tokenize import String
import toml
import pathlib
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)

#Read csv path
config = toml.load(pathlib.Path(__file__).parent / "../config/config.toml")
datapath = pathlib.Path(__file__).parent / f"../../{config['files']['csv']}"


def prepare(datapath: String) -> pd.DataFrame:
    # Load Dataframe
    dataframe = pd.read_csv(datapath)

    # Remove duplicates
    dataframe = dataframe.drop_duplicates(subset=["lesion_id"])

    # Sort values and sort index
    dataframe = dataframe.sort_values("lesion_id").reset_index(drop=True)

    return dataframe

# Put all image names in their corresponding classification list
def class_images(df: pd.DataFrame) -> list:

    # Get classifications
    tags = df.dx.unique().tolist()

    bkl_ids = []
    nv_ids = []
    df_ids = []
    mel_ids = []
    vasc_ids = []
    bcc_ids = []
    akiec_ids = []

    super_list = [bkl_ids, nv_ids, df_ids, mel_ids, vasc_ids, bcc_ids, akiec_ids]

    for i in range(len(tags)):
        super_list[i] = df[df["dx"] == tags[i]][
            "image_id"
        ].values.tolist()

    return super_list


# Preprocess single image
def preprocess(img_path: String) -> np.ndarray:
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img = image.img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    return img


# Preprocess entire list of image ids
def get_preprocessed_images(images_ids: list) -> np.ndarray:
    images = []
    dir1 = config["files"]["dir1"]
    dir2 = config["files"]["dir2"]
    ext = ".jpg"

    # Try to find the images in the first directory, if that doesnt work try the second directory
    for _id in images_ids:
        try:
            images.append(preprocess(dir1 + _id + ext))
        except:
            try:
                # continue
                images.append(preprocess(dir2 + _id + ext))
            except:
                print(dir1 + _id)
                print(dir2 + _id)
                print("Something seriously went wrong")

    return np.vstack(images)


def load_images(super_list):

    # Load your images and preprocess them.
    bkl_images = get_preprocessed_images(super_list[0])
    nv_images = get_preprocessed_images(super_list[1])
    df_images = get_preprocessed_images(super_list[2])
    mel_images = get_preprocessed_images(super_list[3])
    vasc_images = get_preprocessed_images(super_list[4])
    bcc_images = get_preprocessed_images(super_list[5])
    akiec_images = get_preprocessed_images(super_list[6])

    # Make a numpy array for each of the class labels (one hot encoded).
    bkl_labels = np.tile([1, 0, 0, 0, 0, 0, 0], (bkl_images.shape[0], 1))
    nv_labels = np.tile([0, 1, 0, 0, 0, 0, 0], (nv_images.shape[0], 1))
    df_labels = np.tile([0, 0, 1, 0, 0, 0, 0], (df_images.shape[0], 1))
    mel_labels = np.tile([0, 0, 0, 1, 0, 0, 0], (mel_images.shape[0], 1))
    vasc_labels = np.tile([0, 0, 0, 0, 1, 0, 0], (vasc_images.shape[0], 1))
    bcc_labels = np.tile([0, 0, 0, 0, 0, 1, 0], (bcc_images.shape[0], 1))
    akiec_labels = np.tile([0, 0, 0, 0, 0, 0, 1], (akiec_images.shape[0], 1))

    """TODO
    UPSCALING/DOWNSCALING HERE
    """
    
    
    X = np.concatenate(
        [
            bkl_images,
            nv_images,
            df_images,
            mel_images,
            vasc_images,
            bcc_images,
            akiec_images,
        ]
    )

    y = np.concatenate(
        [
            bkl_labels,
            nv_labels,
            df_labels,
            mel_labels,
            vasc_labels,
            bcc_labels,
            akiec_labels,
        ]
    )

    return (X, y)
