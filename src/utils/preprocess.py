import pandas as pd
from skimage import io as sk_io
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm


def preprocess_images(
    csv_file_path, output_dir, root_dir=None, normalization_scheme=None
):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    landmarks_frame = pd.read_csv(csv_file_path)

    # get all files from landmarks_frame
    image_files = landmarks_frame.iloc[:, 0]

    if root_dir:
        image_files = [os.path.join(root_dir, image_file) for image_file in image_files]

    for image_file in tqdm(image_files):

        image = sk_io.imread(image_file).astype(np.float32)  # TODO: float16 or float32?

        # Normalize the image (see )
        if normalization_scheme == "zscore":
            mean = image.mean()
            std = image.std()
            image -= mean
            image /= max(std, 1e-8)

        # Save the image as .npy file
        image_file_name = os.path.basename(image_file).split(".")[0] + ".npy"
        np.save(os.path.join(output_dir, image_file_name), image)
        # TODO: use np.savez_compressed to save disk space / or hdf5 format

        # TODO: resize images to e.g. 1024x1024 and update landmarks accordingly

    # save preprocessing configurations to a file
    with open(os.path.join(output_dir, "preprocessing_config.txt"), "w") as f:
        f.write(f"Normalization scheme: {normalization_scheme}\n")

    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_images(
        csv_file_path="/path/to/CL-Detection2024/Training Set/labels.csv",
        output_dir="/path/to/CL-Detection2024/Training Set/images_preprocessed_zscore/",
        root_dir="/path/to/CL-Detection2024/Training Set/images/",
        normalization_scheme="zscore",
    )
