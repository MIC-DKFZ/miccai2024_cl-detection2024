"""
Project: CL-Detection2024 Challenge Baseline
============================================

This script processes image landmarks and saves them as CSV files. It includes functionality to shuffle
the landmarks list and split it into training, validation, and test sets.
此脚本处理图像关键点并将其保存为CSV文件。打乱关键点列表并将其分为训练集、验证集和测试集。

Email: xiehaoyu2022@email.szu.edu.cn
"""

import os
import random
import pandas as pd
import argparse
from utils.cldetection_utils import check_and_make_dir


def save_landmarks_list_as_csv(
    image_landmarks_list: list, save_csv_path: str, image_dir_path: str
):
    """
    Function to save the landmarks list corresponding to different images in a CSV file
    将不同的图像的关键点以CSV文件保存下来
    :param image_landmarks_list: a list of landmark annotations, each element is an annotation of an image
                                 关键点列表，每一个元素就是一个图片的标注
    :param save_csv_path: CSV file save path
                          CSV文件保存路径
    :param image_dir_path: Directory path where images are stored
                           图像存储的目录路径
    :return: None
    """
    # CSV header | CSV头
    columns = ["image path", "spacing(mm)"]
    for i in range(53):
        columns.extend(["p{}x".format(i + 1), "p{}y".format(i + 1)])
    df = pd.DataFrame(columns=columns)

    # CSV content | CSV内容
    for landmark in image_landmarks_list:
        row_line = [
            os.path.join(image_dir_path, landmark["image file"]),
            landmark["spacing(mm)"],
        ]
        for i in range(53):
            x_position = landmark["p%sx" % (i + 1)]
            y_position = landmark["p%sy" % (i + 1)]
            row_line.extend([x_position, y_position])
        df.loc[len(df.index)] = row_line

    # CSV writer | CSV写入
    df.to_csv(save_csv_path, index=False)


def rename_paths_in_csv(csv_path: str, image_dir_path: str):
    """
    Function to rename the image paths in the CSV file
    将CSV文件中的图像路径重命名
    :param csv_path: Path to the CSV file
                     CSV文件路径
    :param image_dir_path: Directory path where images are stored
                           图像存储的目录路径
    :return: None
    """
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        image_file_path = row["image path"]
        image_file_name = os.path.basename(image_file_path)
        new_image_file_path = os.path.join(image_dir_path, image_file_name)
        df.loc[index, "image path"] = new_image_file_path
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    # # Command line argument parsing | 命令行参数解析
    parser = argparse.ArgumentParser()

    # # Path settings | 路径设置
    parser.add_argument(
        "--images_dir_path",
        type=str,
        default="/path/to/data/Training Set/images/",
        help="Path to the images directory",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="/path/to/data/Training Set/labels.csv",
        help="Path to the labels CSV file",
    )
    parser.add_argument(
        "--split_dir_path",
        type=str,
        default="/path/to/data/Training Set",
        help="Path to the output split directory",
    )

    parser.add_argument(
        "--rename_paths_only",
        type=bool,
        default=False,
        help="Whether to only rename the paths in the CSV files",
    )
    args = parser.parse_args()

    images_dir_path = args.images_dir_path
    labels_path = args.labels_path
    split_dir_path = args.split_dir_path

    check_and_make_dir(split_dir_path)

    train_csv_path = os.path.join(split_dir_path, "train.csv")
    valid_csv_path = os.path.join(split_dir_path, "valid.csv")
    test_csv_path = os.path.join(split_dir_path, "test.csv")

    if not args.rename_paths_only:
        # # Read labels file and convert format | 读取标签文件转换格式
        df = pd.read_csv(labels_path)
        all_image_landmarks_list = df.to_dict(orient="records")

        # # Shuffle the order of the landmark annotations list | 打乱关键点列表的顺序
        random.seed(2024)
        random.shuffle(all_image_landmarks_list)

        # Split the training set, validation set, and test set, and save them as CSV files
        # 划分训练集，验证集和测试集，并以CSV文件形式保存

        print("Train CSV Path:", train_csv_path)
        save_landmarks_list_as_csv(
            image_landmarks_list=all_image_landmarks_list[:300],
            save_csv_path=train_csv_path,
            image_dir_path=images_dir_path,
        )

        print("Valid CSV Path:", valid_csv_path)
        save_landmarks_list_as_csv(
            image_landmarks_list=all_image_landmarks_list[300:350],
            save_csv_path=valid_csv_path,
            image_dir_path=images_dir_path,
        )

        print("Test CSV Path:", test_csv_path)
        save_landmarks_list_as_csv(
            image_landmarks_list=all_image_landmarks_list[350:],
            save_csv_path=test_csv_path,
            image_dir_path=images_dir_path,
        )

    else:

        # Rename the image paths in the CSV files | 重命名CSV文件中的图像路径

        rename_paths_in_csv(train_csv_path, image_dir_path=images_dir_path)
        rename_paths_in_csv(valid_csv_path, image_dir_path=images_dir_path)
        rename_paths_in_csv(test_csv_path, image_dir_path=images_dir_path)
        print("Done!")
