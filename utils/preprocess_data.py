import os
import random

import pandas as pd

import os

train_right = "../dataset/train/right"  # the directory of right images
train_csv_path = "../dataset/train.csv"  # the original train.csv
output_path = "../dataset/extended_train.csv"  # where to save the extended data
# output_path = "../dataset/test_accuracy.csv"
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
    output_path = file_to_transform.split('.')[0] + 'test_accuracy_generated.csv'
    # output_path = file_to_transform.split('.')[0] + 'dataset_generated.csv'
    merged_df.to_csv(output_path)

    return output_path


if __name__ == '__main__':
    #
    # dataset = ExtendedDataset(train_csv_path)
    # dataset.generate_extended_data()
    # dataset.save_to_csv(output_path)

    generate_dataset('../dataset/extended_train.csv', '../dataset/train.csv')

