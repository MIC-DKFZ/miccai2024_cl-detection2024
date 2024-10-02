"""
Project: CL-Detection2024 Challenge Baseline
============================================

Custom functions
自定义函数

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import os
import shutil
import numpy as np
from skimage import io as sk_io
from skimage import draw as sk_draw
import pandas as pd
import time
import subprocess



def check_and_make_dir(dir_path: str) -> None:
    """
    function to create a new folder, if the folder path dir_path in does not exist
    :param dir_path: folder path | 文件夹路径
    :return: None | 无
    """
    if os.path.exists(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError(
                "Error, the provided path (%s) is a file path, not a folder path."
                % dir_path
            )
    else:
        os.makedirs(dir_path)


def calculate_prediction_metrics(result_dict: dict, get_metrics_per_landmark=False, get_std=True):
    """
    function to calculate prediction metrics | 计算评价指标
    :param result_dict: a dict, which stores every image's predict result and its ground truth landmark
                        一个字典，存储每个图像的预测结果及其真实地标
    :param get_metrics_per_landmark: a bool, if True, return metrics per landmark | 如果为True，返回每个地标的指标
    :param get_std: a bool, if True, return standard deviation of metrics | 如果为True，返回指标的标准差
    :return: MRE and 2mm SDR metrics | MRE 和 2mm SDR 指标
    """
    N_CLASSES = 53
    n_landmarks = 0
    sdr_landmarks = 0
    n_landmarks_error = 0
    landmarks_errors = np.zeros((N_CLASSES, len(result_dict)))
    landmarks_sdr = np.zeros((N_CLASSES, len(result_dict)))

    for i, (file_path, landmark_dict) in enumerate(result_dict.items()):
        spacing = landmark_dict["spacing"]
        landmarks, predict_landmarks = landmark_dict["gt"], landmark_dict["predict"]

        # landmarks number
        n_landmarks = n_landmarks + np.shape(landmarks)[0]

        assert N_CLASSES == np.shape(landmarks)[0] == np.shape(predict_landmarks)[0]

        # mean radius error (MRE)
        each_landmark_error = (
            np.sqrt(np.sum(np.square(landmarks - predict_landmarks), axis=1)) * spacing
        )
        n_landmarks_error = n_landmarks_error + np.sum(each_landmark_error)

        # 2mm success detection rate (SDR)
        sdr_landmarks = sdr_landmarks + np.sum(each_landmark_error < 2)

        landmarks_errors[:, i] = each_landmark_error
        landmarks_sdr[:, i] = each_landmark_error < 2

    mean_radius_error = n_landmarks_error / n_landmarks
    sdr = sdr_landmarks / n_landmarks

    if get_std:
        print(f"Mean Radius Error (MRE): {mean_radius_error} +/- {np.std(landmarks_errors)}, 2mm Success Detection Rate (SDR): {sdr} +/- {np.std(landmarks_sdr)}")
        mean_radius_error = (mean_radius_error, np.std(landmarks_errors))
        sdr = (sdr, np.std(landmarks_sdr))
    else:
        print(
            "Mean Radius Error (MRE): {}, 2mm Success Detection Rate (SDR): {}".format(
                mean_radius_error, sdr
            )
        )
    # Return results
    if get_metrics_per_landmark:
        landmarks_errors_std = np.std(landmarks_errors, axis=1)
        landmarks_sdr_std = np.std(landmarks_sdr, axis=1)
        landmarks_errors = np.mean(landmarks_errors, axis=1)  # calculate mean
        landmarks_sdr = np.mean(landmarks_sdr, axis=1)      
        # create a DataFrame to store the results with a column for each landmark
        df = pd.DataFrame(
            {
                "Mean Radius Error": landmarks_errors,
                "Mean Radius Error Std": landmarks_errors_std,
                "2mm Success Detection Rate": landmarks_sdr,
                "2mm Success Detection Rate Std": landmarks_sdr_std,
            },
            index=[f"Landmark {i+1}" for i in range(N_CLASSES)]
        )
        return (mean_radius_error, sdr), df
    else:
        return mean_radius_error, sdr


def save_metrics_to_txt(model_name, mre, sdr, mre_std=True, sdr_std=True, weights="", filename="results.txt"):
    with open(filename, "a") as file:
        file.write(f"Model: {model_name}, weights={weights} \n")
        file.write(f"Mean Radius Error (MRE): {mre}\n")
        if mre_std is not None:
            file.write(f"Mean Radius Error (MRE) Std: {mre_std}\n")
        file.write(f"2mm Success Detection Rate (SDR): {sdr}\n")
        if sdr_std is not None:
            file.write(f"2mm Success Detection Rate (SDR) Std: {sdr_std}\n")
        file.write("\n")


def visualize_prediction_landmarks(result_dict: dict, save_image_dir: str):
    """
    function to visualize prediction landmarks | 可视化预测结果
    :param result_dict: a dict, which stores every image's predict result and its ground truth landmark
                       一个字典，存储每个图像的预测结果及其真实地标
    :param save_image_dir: the folder path to save images | 一个字典，存储每个图像的预测结果及其真实地标
    :return: None | 无
    """
    for file_path, landmark_dict in result_dict.items():
        landmarks, predict_landmarks = landmark_dict["gt"], landmark_dict["predict"]

        image = sk_io.imread(file_path)
        image_shape = np.shape(image)[:2]

        for i in range(np.shape(landmarks)[0]):
            landmark, predict_landmark = landmarks[i, :], predict_landmarks[i, :]
            # ground truth landmark
            radius = 7
            rr, cc = sk_draw.disk(
                center=(int(landmark[1]), int(landmark[0])),
                radius=radius,
                shape=image_shape,
            )
            image[rr, cc, :] = [0, 255, 0]
            # model prediction landmark
            rr, cc = sk_draw.disk(
                center=(int(predict_landmark[1]), int(predict_landmark[0])),
                radius=radius,
                shape=image_shape,
            )
            image[rr, cc, :] = [255, 0, 0]
            # the line between gt landmark and prediction landmark
            line_width = 5
            rr, cc, value = sk_draw.line_aa(
                int(landmark[1]),
                int(landmark[0]),
                int(predict_landmark[1]),
                int(predict_landmark[0]),
            )
            for offset in range(line_width):
                offset_rr, offset_cc = np.clip(
                    rr + offset, 0, image_shape[0] - 1
                ), np.clip(cc + offset, 0, image_shape[1] - 1)
                image[offset_rr, offset_cc, :] = [255, 255, 0]

        filename = os.path.basename(file_path)
        sk_io.imsave(os.path.join(save_image_dir, filename), image)


def log_gpu_memory(log_file):
    """
    Function to log GPU memory usage using nvidia-smi.
    Writes the log to a specified file.
    """
    with open(log_file, 'w') as f:
        print("File opened successfully. Writing header...")  # Debugging message
        f.write('timestamp,memory.used [MiB]\n')  # CSV header
        f.flush()  # Force writing the header to the file immediately
        while True:
            try:
                # Run nvidia-smi and query memory usage
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=timestamp,memory.used', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                # Write the output to the log file
                f.write(result.stdout)
                f.flush()  # Force writing the header to the file immedi
                time.sleep(1)  # Log every 1 second
            except Exception as e:
                print(f"Error while logging GPU memory: {e}")
                break



def compute_auc(log_file):
    """
    Function to compute the area under the curve (AUC) for GPU memory usage.
    """
    df = pd.read_csv(log_file)

    # Convert the timestamp column to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate time differences between successive rows in seconds
    df['time_diff_sec'] = df['timestamp'].diff().dt.total_seconds()

    # Fill the first time_diff with 0 as there's no previous entry
    df['time_diff_sec'].fillna(0, inplace=True)

    # Compute the area under the curve (MiB·s)
    df['auc_contrib'] = df['memory.used [MiB]'] * df['time_diff_sec']

    # Sum all the contributions to get total area under curve (MiB·s)
    total_auc = df['auc_contrib'].sum()

    return total_auc
