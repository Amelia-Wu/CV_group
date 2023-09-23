import os
import random

import numpy as np
import pandas as pd

import os

from keras_preprocessing.image import load_img
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

train_right = "../dataset/train/right"  # the directory of right images
train_csv_path = "../dataset/train.csv"  # the original train.csv
output_path = "../dataset/extended_train.csv"  # where to save the extended data
right_img_num = 20  # how many right images for each left image

def get_filenames_in_directory(directory_path):
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

class ExtendedDataset:
    """
    This class is used to generate the extended dataset.
    Read the original train.csv, and generate a new csv file with 20 right images for each left image.
    """

    def __init__(self, data_filepath):
        self.data = pd.read_csv(data_filepath)
        self.available_filenames = get_filenames_in_directory(train_right)
        self.extended_data = None

    def generate_extended_data(self):
        extended_data = []
        for index, row in self.data.iterrows():
            left_image = row['left']
            right_image = row['right']

            # 从文件名列表中移除ground truth
            available_choices = [f for f in self.available_filenames if f != right_image]

            additional_images = random.sample(available_choices, right_img_num-1)

            additional_images = [f.split('.')[0] for f in additional_images]

            all_right_images = [right_image] + additional_images

            extended_data.append([left_image]+ all_right_images)

        columns = ['left'] + ['c' + str(i) for i in range(right_img_num)]
        self.extended_data = pd.DataFrame(extended_data, columns=columns)


    def save_to_csv(self, save_path):
        if self.extended_data is not None:
            self.extended_data.to_csv(save_path, index=False)
        else:
            print("Please generate the extended data first.")


def generate_dataset(file_to_transform, ground_truth_file):
    """
    Generate dataset from the format of the original extended_train.csv/test_candidates.csv into the format of data pairs.
    file_to_transform: the file to transform (like extended_train.csv/ test_candidates.csv)
    ground_truth_file: the ground truth file (like train.csv), we will use it to add labels to the generated dataset.
    :return: Path to the generated dataset
    """
    df = pd.read_csv(file_to_transform, index_col=None)
    melted_df = df.melt(id_vars=["left"], value_name="right")
    # 删除临时列variable
    melted_df.drop(columns=['variable'], inplace=True)

    # 读取ground_truth_file并设置left为index
    ground_truth_df = pd.read_csv(ground_truth_file)

    # 使用join方法将melted_df和ground_truth_df进行合并
    merged_df = melted_df.join(ground_truth_df, rsuffix='_ground_truth', how='left')

    # 比较melted_df中的right列和ground_truth_df中的right列是否匹配
    merged_df['label'] = (merged_df['right'] == merged_df['right_ground_truth']).astype(int)

    # 删除临时列right_ground_truth
    merged_df.drop(columns=['right_ground_truth'], inplace=True)
    # 删除临时列 left_ground_truth
    merged_df.drop(columns=['left_ground_truth'], inplace=True)

    # 保存到新的CSV文件
    output_path = file_to_transform.split('.')[0] + 'dataset_generated.csv'
    merged_df.to_csv(output_path)

    return output_path


import tensorflow as tf




class ImagePairsDataset:
    """
    # 示例使用：
    # train_image_pairs = [('path/to/image1a.jpg', 'path/to/image1b.jpg'), ...]
    # train_labels = [1, ...]
    # train_dataset = ImagePairsDataset(train_image_pairs, train_labels).get_dataset()
    """
    def __init__(self, image_pairs, labels, batch_size=32, shuffle=True):
        self.image_pairs = image_pairs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = self._create_dataset()

    def _load_img(self, img_path1, img_path2, label):
        try:
            img1 = tf.io.read_file(img_path1)
            img1 = tf.image.decode_jpeg(img1, channels=3)
            img1 = tf.image.resize(img1, [224, 224])
            img1 = img1 / 255.0
        except Exception as e:
            print(f"Error processing {img_path1}: {e}")
            return None, label

        try:
            img2 = tf.io.read_file(img_path2)
            img2 = tf.image.decode_jpeg(img2, channels=3)
            img2 = tf.image.resize(img2, [224, 224])
            img2 = img2 / 255.0
        except Exception as e:
            print(f"Error processing {img_path2}: {e}")
            return None, label

        return tf.subtract(img1, img2), label

    def _create_dataset(self):
        img1_paths = [pair[0] for pair in self.image_pairs]
        img2_paths = [pair[1] for pair in self.image_pairs]

        dataset = tf.data.Dataset.from_tensor_slices((img1_paths, img2_paths, self.labels))
        dataset = dataset.map(self._load_img, num_parallel_calls=tf.data.AUTOTUNE)  #TODO: map干啥的


        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.labels), seed=42)

        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_dataset(self):
        return self.dataset


def get_data_pairs(csv_path, img_path_prefix='../dataset/train', img_suffix='.jpg', val_ratio=0.2):
    """
    Read image pairs from CSV files, process their paths, disrupt the data, and split into training and validation datasets.

    参数：
    - csv_path: The path to the CSV file containing the image pairs and labels.
    - img_path_prefix: The prefix of the image file path.
    - img_suffix: The suffix of the image file name.
    - val_ratio: Proportion of validation sets.

    返回：
    - train_dataset
    - val_dataset
    """
    # 从CSV读取数据
    data = pd.read_csv(csv_path)
    image_pairs = list(zip(data['left'].values, data['right'].values))
    labels = data['label'].values.tolist()
    # 创建一个OneHotEncoder对象
    encoder = OneHotEncoder(sparse=False)

    # 将标签列表转换为NumPy数组并重塑以匹配OneHotEncoder的输入要求
    labels = np.array(labels).reshape(-1, 1)

    # 进行独热编码
    one_hot_labels = encoder.fit_transform(labels)

    # 处理图像路径
    image_pairs = [(img_path_prefix + "/left/" + img1 + img_suffix, img_path_prefix + "/right/" + img2 + img_suffix) for img1, img2 in
                   image_pairs]

    # 打乱并分割数据
    train_image_pairs, val_image_pairs, train_labels, val_labels = train_test_split(
        image_pairs, one_hot_labels, test_size=val_ratio, random_state=42)

    # 创建tf.data.Dataset对象
    train_dataset = ImagePairsDataset(train_image_pairs, train_labels).get_dataset()
    val_dataset = ImagePairsDataset(val_image_pairs, val_labels, shuffle=False).get_dataset()

    return train_dataset, val_dataset


if __name__ == '__main__':
    pass
    # dataset = ExtendedDataset(train_csv_path)
    # dataset.generate_extended_data()
    # dataset.save_to_csv(output_path)

    # generate_dataset('../dataset/extended_train.csv', '../dataset/train.csv')

    train_dataset, val_dataset = get_data_pairs('E:/3Melbourne_uni_2023_S2/cv/assignment/Group/code/utils/dataset_generated.csv')








# ('../dataset/train/left/aaa.jpg', '../dataset/train/right/osr.jpg')

