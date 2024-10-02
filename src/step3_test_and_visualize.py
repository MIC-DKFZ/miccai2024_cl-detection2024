# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

"""
Project: CL-Detection2024 Challenge Baseline
============================================

This script implements cephalometric landmarks prediction on X-Ray images using a UNet-based heatmap approach.
It includes functionality for model testing, calculating MRE and SDR metrics, and visualizing the results.
此脚本实现了基于UNet热图的X射线图像头影关键点预测，包括模型测试、计算MRE和SDR指标以及结果可视化。

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
from skimage import transform
from skimage import io as sk_io
import os
import multiprocessing
import time
import warnings

warnings.filterwarnings("ignore")

from utils.model import load_model
from utils.cldetection_utils import (
    check_and_make_dir,
    calculate_prediction_metrics,
    visualize_prediction_landmarks,
    save_metrics_to_txt,
    log_gpu_memory,
    compute_auc
)

import yaml

import numpy as np
from PIL import Image


def save_heatmap_as_image(heatmap, filename):
    """
    Saves a single-channel heatmap as an image.

    :param heatmap: A 2D NumPy array representing the heatmap (single-channel).
    :param filename: The filename (including path) to save the image.
    """
    # Normalize the heatmap to 0-255
    heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap_normalized = (heatmap_normalized * 255).astype(np.uint8)

    # Convert the normalized heatmap to a PIL image
    heatmap_image = Image.fromarray(heatmap_normalized)

    # Save the image
    heatmap_image.save(f"/path/to/projects/2024_MICCAI24_CL-Detection2024_challenge/heatmaps/{filename}")



def main(config):
    """
    Main function for model test and visualization.
    主函数，用于模型测试和可视化。
    :param config: Configuration object containing various configuration parameters.
                   包含各种配置参数的配置对象
    """

    # Configs
    if config.test_csv_path is None:
        config.test_csv_path = f"/path/to/data/CL-Detection/CL-Detection2024 Accessible Data/Training Set/{config.split}.csv"
    if config.save_metrics_path is None:
        config.save_metrics_path = f"/path/to/projects/2024_MICCAI24_CL-Detection2024_challenge/experiments/{config.experiment_name}/metrics/"
    if config.load_weight_path is None:
        config.load_weight_path = f"/path/to/projects/2024_MICCAI24_CL-Detection2024_challenge/experiments/{config.experiment_name}/final_model.pt"
    if config.save_image_dir is None:
        config.save_image_dir = f"/path/to/projects/2024_MICCAI24_CL-Detection2024_challenge/experiments/{config.experiment_name}/visualize/"
        
    # GPU device | GPU 设备
    gpu_id = config.cuda_id
    # device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:{}'.format(gpu_id))

    # start monitoring runtime and gpu memory 
    log_file = os.path.join(config.save_metrics_path, 'gpu_memory_log.csv')

    # Start logging GPU memory in a separate process
    print("Starting GPU memory logging...")
    log_process = multiprocessing.Process(target=log_gpu_memory, args=(log_file,))
    log_process.start()
    start_time = time.time()
    
    try:

        # Load model | 加载模型
        if config.model_name is not None:
            model = load_model(model_name=config.model_name)
        else:
            # load from hydra config yaml
            model_configs = yaml.load(
                open(f"/path/to/projects/2024_MICCAI24_CL-Detection2024_challenge/experiments/{config.experiment_name}/.hydra/config.yaml", "r")
                , Loader=yaml.FullLoader)
            model = load_model(model_name=model_configs["model_name"], **model_configs.get('model_kwargs', {}))
            #model = load_model(model_name="UNet_18")
        model.load_state_dict(torch.load(config.load_weight_path, map_location=device))
        model = model.to(device)
        print("model loaded to gpu")
        # Load test.csv | 加载测试数据集
        df = pd.read_csv(config.test_csv_path)

        # Test result dict | 测试结果字典
        test_result_dict = {}

        # Test mode | 测试模式
        with torch.no_grad():
            model.eval()
            # Test all images | 测试所有图片
            for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
                image_file_path, spacing = str(df.iloc[index, 0]), float(df.iloc[index, 1])
                landmarks = df.iloc[index, 2:].values.astype("float")
                landmarks = landmarks.reshape(-1, 2)

                # Load image array | 加载图像数组
                image = sk_io.imread(image_file_path)
                h, w = image.shape[:2]
                new_h, new_w = config.image_height, config.image_width

                # Preprocessing image for model input | 为模型输入预处理图像
                image = transform.resize(
                    image, (new_h, new_w), mode="constant", preserve_range=False
                )
                #image = np.transpose(image, (2, 0, 1))

                image = image[:, :, 0]
                image =  np.expand_dims(image, axis=0)

                image = torch.from_numpy(image[np.newaxis, :, :, :]).float().to(device)

                if config.test_transform == "zscore":
                    mean = image.mean()
                    std = image.std()
                    image -= mean
                    image /= max(std, 1e-8)  
                
                print(torch.max(image))
                print(torch.min(image))

                # Predict heatmap | 预测热图
                heatmap = model(image)
                
                #if model_configs.get("final_activation", None) == "sigmoid":
                #    heatmap = torch.sigmoid(heatmap)

                # Transfer to landmarks | 转换为地标
                heatmap = np.squeeze(heatmap.cpu().numpy())
                predict_landmarks = []
                for i in range(np.shape(heatmap)[0]):
                    landmark_heatmap = heatmap[i, :, :]
                    #save_heatmap_as_image(landmark_heatmap, f'heatmap_{index}_{i}.png')
                    yy, xx = np.where(landmark_heatmap == np.max(landmark_heatmap))
                    # There may be multiple maximum positions, and a simple average is performed as the final result
                    # 可能存在多个最大位置，并且进行简单平均以作为最终结果
                    x0, y0 = np.mean(xx), np.mean(yy)
                    # Zoom to original image size | 缩放到原始图像大小
                    x0, y0 = x0 * w / new_w, y0 * h / new_h
                    # Append to predict landmarks | 添加到预测地标
                    predict_landmarks.append([x0, y0])

                test_result_dict[image_file_path] = {
                    "spacing": spacing,
                    "gt": np.asarray(landmarks),
                    "predict": np.asarray(predict_landmarks),
                }
                del heatmap, predict_landmarks, image
                torch.cuda.empty_cache()
        
        print("Finished predictions.")

        # Calculate prediction metrics | 计算预测指标
        prediction_metrics = calculate_prediction_metrics(test_result_dict)
        if not os.path.exists(config.save_metrics_path):
            os.makedirs(config.save_metrics_path)
        save_metrics_to_txt(
            model_name=config.model_name,
            mre=prediction_metrics[0],
            sdr=prediction_metrics[1],
            filename=os.path.join(config.save_metrics_path, f"{config.split}_metrics.txt"),
        )
        # save test_result_dict as csv
        # df = pd.DataFrame(test_result_dict)
        # df.to_csv(config.save_csv_path, index=False)

    finally:
        # Once the script is done, terminate the logging process
        print("Stopping GPU memory logging...")
        log_process.terminate()
        # Give it a second to terminate cleanly
        end_time = time.time()
        time.sleep(1)

    # Compute the area under the curve
    print("Computing GPU memory usage (MiB·s)...")
    total_auc = compute_auc(log_file)
    print(f"Total GPU Memory Usage (MiB·s): {total_auc}")
    print("Runtime: " + str(end_time-start_time))
    
    # Visualize prediction landmarks | 可视化预测地标
    if config.save_image:
        check_and_make_dir(config.save_image_dir)
        visualize_prediction_landmarks(test_result_dict, config.save_image_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    experiment_name = "experiment001"

    parser.add_argument("--experiment_name", type=str, default=experiment_name)
    parser.add_argument("--split", type=str, default="test")
    
    parser.add_argument("--test_csv_path", type=str,default=None)
    parser.add_argument("--save_csv_path",type=str,default=None)
    parser.add_argument("--save_metrics_path",type=str,default=None,)
    parser.add_argument("--save_image", type=bool, default=True)
    parser.add_argument("--save_image_dir",type=str,default=None,)
    parser.add_argument("--load_weight_path",type=str,default=None,)

    parser.add_argument("--test_transform",type=str,default="zscore",)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--image_height", type=int, default=1024)

    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--model_name", type=str, default=None)

    experiment_config = parser.parse_args()
    main(experiment_config)
