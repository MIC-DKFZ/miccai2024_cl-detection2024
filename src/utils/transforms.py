"""
Project: CL-Detection2024 Challenge Baseline
============================================

Data transform
数据转换

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import random
import numpy as np
from skimage import transform as sk_transform
import torch
from torchvision import tv_tensors
import torchvision.transforms.functional as F
import torchvision.transforms as T
import cv2
from skimage import transform
from scipy.ndimage import map_coordinates, gaussian_filter

import random
import numpy as np
from skimage import transform as sk_transform
import torch
from torchvision import tv_tensors


def generate_2d_gaussian_heatmap(
    heatmap: np.ndarray, center: tuple, sigma=20, radius=50
):
    """
    function to generate 2d gaussian heatmap.
    生成二维高斯热图。
    :param heatmap: heatmap array | 传入进来赋值的高斯热图
    :param center: a tuple, like (x0, y0) | 中心的坐标
    :param sigma: gaussian distribution sigma value | 高斯分布的sigma值
    :param radius: gaussian distribution radius | 高斯分布考虑的半径范围
    :return: heatmap array | 热图
    """
    x0, y0 = center
    xx, yy = np.ogrid[-radius : radius + 1, -radius : radius + 1]

    # generate gaussian distribution | 生成高斯分布
    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    # valid range | 有效范围
    height, width = np.shape(heatmap)
    left, right = min(x0, radius), min(width - x0, radius + 1)
    top, bottom = min(y0, radius), min(height - y0, radius + 1)

    # assign operation | 赋值操作
    masked_heatmap = heatmap[y0 - top : y0 + bottom, x0 - left : x0 + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    # the np.maximum function is used to avoid aliasing of multiple landmarks on the same heatmap
    # 使用 np.maximum 函数来避免在同一个热图上对多个地标进行别名处理
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    调整样本中的图像大小以匹配给定大小。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        h, w = image.shape[:2]
        new_h, new_w = int(self.output_size[0]), int(self.output_size[1])

        image = sk_transform.resize(
            image, (new_h, new_w), mode="constant", preserve_range=False
        )
        landmarks = landmarks * [new_w / w, new_h / h]

        return {"image": image, "landmarks": landmarks}


class RandomHorizontalFlip(object):
    """
    Flip randomly the image in a sample.
    随机水平翻转一个样本的图像
    """

    def __init__(self, p):
        assert isinstance(p, float)
        self.prob = p

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        if random.random() < self.prob:
            _, w = image.shape[:2]
            landmarks[:, 0] = w - landmarks[:, 0]
            image = image[:, ::-1, :].copy()

        return {"image": image, "landmarks": landmarks}

class RandomRotation(object):
    """
    Rotate the image in a sample by a random angle.
    随机旋转样本中的图像。
    """

    def __init__(self, degrees):
        assert isinstance(degrees, (int, float))
        self.degrees = degrees

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        # Randomly select an angle within the specified range
        angle = random.uniform(-self.degrees, self.degrees)

        # Get image dimensions
        h, w = image.shape[:2]

        # Calculate the center of the image
        center = (w / 2, h / 2)

        # Compute the rotation matrix
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply the rotation to the image
        rotated_image = cv2.warpAffine(image, rot_matrix, (w, h))
        rotated_image = np.expand_dims(rotated_image, axis=-1)
        # Convert landmarks to homogeneous coordinates (add a column of ones)
        ones = np.ones((landmarks.shape[0], 1))
        landmarks_homogeneous = np.hstack([landmarks, ones])

        # Rotate the landmarks using the rotation matrix
        rotated_landmarks = rot_matrix.dot(landmarks_homogeneous.T).T

        # Return the rotated image and updated landmarks
        return {"image": rotated_image, "landmarks": rotated_landmarks[:, :2]}
    

class ElasticTransform(object):
    """
    Apply elastic deformation to both image and landmarks.
    对图像和标记同时进行弹性变形
    """
    
    def __init__(self, alpha, sigma, p=0.5):
        """
        Args:
            alpha (float): Scaling factor that controls the intensity of the deformation.
            sigma (float): Standard deviation of the Gaussian filter.
            p (float): Probability of applying the transform.
        """
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < self.p:
            # Apply elastic deformation to image
            image, displacement_field = self.elastic_transform(image, self.alpha, self.sigma)
            
            # Apply the same displacement field to landmarks
            landmarks = self.transform_landmarks(landmarks, displacement_field)
        
        return {'image': image, 'landmarks': landmarks}
    
    def elastic_transform(self, image, alpha, sigma):
        """Elastic deformation of images as described in [Simard2003]."""
        random_state = np.random.RandomState(None)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        displacement_field = np.stack((dx, dy), axis=-1)
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        distored_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        
        return distored_image, displacement_field
    
    def transform_landmarks(self, landmarks, displacement_field):
        """Apply the computed displacement field to the landmarks."""
        for i in range(len(landmarks)):
            # Round the coordinates to use in the displacement field
            x, y = int(round(landmarks[i, 0])), int(round(landmarks[i, 1]))

            # Ensure coordinates are within image boundaries
            if x >= 0 and x < displacement_field.shape[1] and y >= 0 and y < displacement_field.shape[0]:
                # Apply the displacement to each landmark coordinate
                dx, dy = displacement_field[y, x]
                landmarks[i, 0] += dx
                landmarks[i, 1] += dy

        return landmarks

class ToTensor(object):
    """
    Convert image array in sample to Tensors and generate heatmaps for landmarks.
    """

    def __init__(self, sigma=8):
        self.sigma = sigma  # Initialize with default sigma

    def set_sigma(self, sigma):
        """
        Dynamically update the sigma value.
        """
        self.sigma = sigma

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        # generate all landmarks' heatmap | 生成所有地标的热图
        h, w = image.shape[:2]
        n_landmarks = np.shape(landmarks)[0]
        heatmap = np.zeros((n_landmarks, h, w))
        for i in range(n_landmarks):
            center = (int(landmarks[i, 0] + 0.5), int(landmarks[i, 1] + 0.5))
            heatmap[i, :, :] = generate_2d_gaussian_heatmap(
                heatmap[i, :, :], center, sigma=self.sigma, radius=20
            )

        # swap color axis because numpy image: H x W x C but torch image: C X H X W
        image = image.transpose((2, 0, 1))

        # return TVTensor instead of numpy array for further transformations
        return tv_tensors.Image(image), tv_tensors.Mask(heatmap)
        # return image, heatmap


class RandomTranslation(object):
    """
    Randomly translate the image and adjust landmarks accordingly by a proportion of the image size.
    """

    def __init__(self, translate_range=(0, 0.1), p=0.5):
        """
        Args:
            translate_range (tuple): Tuple indicating the range (min, max) of proportion to translate the image.
                                     Proportion is relative to the image width and height.
                                     e.g., (0, 0.1) means up to 10% of the image size in any direction.
        """
        assert isinstance(translate_range, tuple) and len(translate_range) == 2, "translate_range must be a tuple of (min, max)"
        self.translate_range = translate_range
        self.p = p

    def __call__(self, sample):
        image, heatmap = sample  # Assuming sample is a tuple (image, heatmap)

        if random.random() < self.p:
            h, w = image.shape[1:]  # Get image height and width assuming image shape (C, H, W)

            # Randomly choose translation proportions for x and y within the given range
            translate_x_prop = np.random.uniform(-self.translate_range[1], self.translate_range[1])
            translate_y_prop = np.random.uniform(-self.translate_range[1], self.translate_range[1])
            
            # Calculate the actual translation in pixels
            translate_x = translate_x_prop * w
            translate_y = translate_y_prop * h

            # Create the translation matrix
            translation_matrix = transform.AffineTransform(translation=(translate_x, translate_y))

            # Initialize arrays for translated image and heatmap
            translated_image = torch.zeros_like(image)
            translated_heatmap = torch.zeros_like(heatmap)

            # Convert each channel to numpy, apply warp, and convert back to tensor
            for c in range(image.shape[0]):  # Loop through channels of the image
                image_np = image[c].cpu().numpy()  # Convert to numpy array
                translated_image_np = transform.warp(image_np, translation_matrix, mode='constant', preserve_range=True)
                translated_image[c] = torch.from_numpy(translated_image_np).type_as(image)

            for c in range(heatmap.shape[0]):  # Loop through channels of the heatmap
                heatmap_np = heatmap[c].cpu().numpy()  # Convert to numpy array
                translated_heatmap_np = transform.warp(heatmap_np, translation_matrix, mode='constant', preserve_range=True)
                translated_heatmap[c] = torch.from_numpy(translated_heatmap_np).type_as(heatmap)

            image = translated_image
            heatmap = translated_heatmap

        return image, heatmap
    

class GaussianNoise(object):
    """
    Add Gaussian noise with a variance uniformly sampled from the given range.
    """

    def __init__(self, noise_variance=(0, 0.1)):
        self.std = np.random.uniform(noise_variance[0], noise_variance[1])

    def __call__(self, image):
        # image: torch.Tensor
        image = image + self.std * torch.randn_like(image)

        return image


class ZScoreNormalization(object):
    """
    Apply Z-score normalization to the image while keeping landmarks unchanged.
    """

    def __call__(self, image):

        mean = image.mean()
        std = image.std()
        image -= mean
        image /= max(std, 1e-8)

        return image
    

class Grayscale(object):

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        image = image[:, :, 0]
        image =  np.expand_dims(image, axis=-1)

        return {"image": image, "landmarks": landmarks}