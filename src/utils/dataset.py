"""
Project: CL-Detection2024 Challenge Baseline
============================================

Data Reading
数据读取

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import pandas as pd
from skimage import io as sk_io
from torch.utils.data import Dataset
import os
import numpy as np


class CephXrayDataset(Dataset):
    def __init__(
        self,
        csv_file_path,
        root_dir=None,
        transform=None,
        image_transform=None,
        load_preprocessed=False,
        *args,
        **kwargs
    ):
        """
        Initialize the CephXrayDataset with a CSV file containing image file paths and landmarks.
        使用包含图像文件路径和关键点的CSV文件初始化CephXrayDataset。
        Args:
            csv_file_path (str): Path to the CSV file containing image paths and landmarks.
                                 包含图像路径和关键点的CSV文件路径。
            root_dir (str): Directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
                                            可选的转换应用于样本。
            image_transform (callable, optional): Optional transform to be applied on the image only.
            load_preprocessed (bool): Whether to load preprocessed images (saved as .npy files).
        """
        super().__init__(*args, **kwargs)
        self.landmarks_frame = pd.read_csv(csv_file_path)
        self.transform = transform
        self.image_transform = image_transform
        self.root_dir = root_dir
        self.load_preprocessed = load_preprocessed
        if self.load_preprocessed:
            print(f'Loading preprocessed image from {self.root_dir}...')

    def __getitem__(self, index):
        """
        Get a sample from the dataset at the specified index.
        获取指定索引的数据集样本。
        Args:
            index (int): Index of the sample to be fetched.
                         要获取的样本索引。
        Returns:
            sample (dict): A dictionary containing the image and its landmarks.
                           包含图像及其关键点的字典。
        """
        if self.load_preprocessed:
            # check if self.root_dir is None
            if self.root_dir is None:
                raise ValueError("root_dir must be provided if load_preprocessed is True.")
            image_file_name = os.path.basename(self.landmarks_frame.iloc[index, 0]).split(".")[0] + ".npy"
            image_file_path = os.path.join(self.root_dir, image_file_name)
            image = np.load(image_file_path)

        else:
            image_file_path = str(self.landmarks_frame.iloc[index, 0])
            if self.root_dir:
                image_file_path = os.path.join(self.root_dir, image_file_path)
            image = sk_io.imread(image_file_path)

        landmarks = self.landmarks_frame.iloc[index, 2:].values.astype("float")
        landmarks = landmarks.reshape(-1, 2)

        # Apply the transform if provided | 如果提供转换则应用
        if self.transform is not None:
            sample = self.transform({"image": image, "landmarks": landmarks})
            image, landmarks = sample
        else:
            sample = {"image": image, "landmarks": landmarks}

        # Apply the image transform to the image only
        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, landmarks
        #return {"image": image, "landmarks": landmarks}

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        返回数据集中样本的总数。
        """
        return len(self.landmarks_frame)



