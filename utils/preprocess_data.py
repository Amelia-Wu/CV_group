import os
import random

import pandas as pd

import os

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




if __name__ == '__main__':

    dataset = ExtendedDataset(train_csv_path)
    dataset.generate_extended_data()
    dataset.save_to_csv(output_path)

