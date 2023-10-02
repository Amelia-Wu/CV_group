import time

import numpy as np
import os

from sklearn.svm import SVC

from utils.npz_file_reader import NPZReader
import pandas as pd
from keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

#TODO:
# 1. 试试 ResNet50, VGG19
# 2. 试试低级特征（从其他层捕获特征）, 不用全连接层时，可以将target_size设置为原始图片大小

target_size = (224, 224)  # 因为全连接层fc1，所以size固定为224
# target_size = (200, 245)

class VggFeatureExtractor:
    def __init__(self):
        # Load VGG16 model + higher level layers
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img_path):
        # Load image with size (224, 224)
        img = image.load_img(img_path, target_size=target_size)
        # Convert image to array
        img_array = image.img_to_array(img)
        # Convert to a batch of size (1, 224, 224, 3)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        # Preprocess the input for VGG16 model
        preprocessed_img = preprocess_input(expanded_img_array)
        # Get features
        features = self.model.predict(preprocessed_img)
        return features

    def save_features(self, directory, output_file):
        feature_list = []
        filenames = []

        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                feature = self.extract(os.path.join(directory, filename))
                feature_list.append(feature)
                filenames.append(filename)

        # Save the features to a file
        np.savez(output_file, features=feature_list, filenames=filenames)

def load_specific_feature(npz_file, target_filename):
    # Load the .npz file
    data = np.load(npz_file)

    # Get the list of filenames and features
    filenames = data['filenames']
    features = data['features']

    # Find the index of the target filename
    index = np.where(filenames == target_filename)[0][0]

    # Return the corresponding feature
    return features[index]


import matplotlib.pyplot as plt


def plot_feature(feature, title=None):
    # Reshape feature to a square shape for visualization
    size = int(feature.shape[1] ** 0.5)
    reshaped_feature = feature.reshape(size, size)

    plt.imshow(reshaped_feature, cmap='viridis')

    # Remove axis
    plt.axis('off')
    plt.colorbar()  # represent the eigenvalues with a color bar
    if title:
        plt.title(title+' feature map')
    plt.show()


import numpy as np


def cosine_similarity(array1, array2):
    dot_product = np.dot(array1, array2.T)
    norm_array1 = np.linalg.norm(array1)
    norm_array2 = np.linalg.norm(array2)

    cosine_similarity = dot_product / (norm_array1 * norm_array2)

    return cosine_similarity

def compute_cosine_similarity_for_submission(extended_train_path, left_features_path, right_features_path, output_path):
    """
    Compute the cosine similarity for the left and right images and format the output similar to sample submission.
    """
    # 读取extended_train_path的每一行，每一行第一列是left image的文件名，后面20列是20个right image的文件名
    # 从left_features_path中读取left image的特征， 从right_features_path中读取right image的特征
    # 计算left image和所有right image分别的cosine similarity， 用值替换right image的文件名(对应行对应列的值)
    # Load the data
    extended_train_df = pd.read_csv(extended_train_path)
    # 列数
    num_cols = extended_train_df.shape[1]

    # Load the .npz file
    npzReader = NPZReader(left_features_path, right_features_path)

    # Iterate over each row in the extended_train_df
    for index, row in extended_train_df.iterrows():
        print("Processing row: ", index)
        start = time.time()
        left_image = row[0]+'.jpg'
        left_feature = npzReader.get_feature(left_image)

        for i in range(1, num_cols):  # Assuming there are 20 right images per row
            right_image = row[i]+'.jpg'
            right_feature = npzReader.get_feature(right_image)

            # Compute cosine similarity
            similarity = cosine_similarity(left_feature, right_feature)[0][0]

            # Replace the filename with the similarity value
            extended_train_df.iloc[index, i] = similarity
        end = time.time()
        print(f"Time elapsed: {end-start}; ")

    # save the extended_train_df to a csv file
    extended_train_df.to_csv(output_path, index=False)

    return extended_train_df

def svm_for_submission(extended_train_path, left_features_path, right_features_path):

    extended_train_df = pd.read_csv(extended_train_path)
    # 列数
    num_cols = extended_train_df.shape[1]

    # Load the .npz file
    npzReader = NPZReader(left_features_path, right_features_path)


    features = []
    labels = []
    # Iterate over each row in the extended_train_df
    for index, row in extended_train_df.iterrows():
        print("Processing row: ", index)
        start = time.time()
        left_image = row[0]+'.jpg'
        left_feature = npzReader.get_feature(left_image)

        for i in range(1, num_cols):  # Assuming there are 20 right images per row
            right_image = row[i]+'.jpg'
            right_feature = npzReader.get_feature(right_image)

            # combine left_feature and right_feature
            feature = np.concatenate((left_feature, right_feature), axis=1)
            feature_squeezed = np.squeeze(feature)
            features.append(feature_squeezed)

            # label
            if i == 1:
                labels.append(1)
            else:
                labels.append(0)
        end = time.time()
        print(f"Time elapsed: {end-start}; ")

    return features, labels


def svm_predict(extended_train_path, left_features_path, right_features_path, features, labels, output_path):
    svm = SVC(probability=True, random_state=42)
    print("start fitting")
    svm.fit(features, labels)

    extended_train_df = pd.read_csv(extended_train_path)
    # 列数
    num_cols = extended_train_df.shape[1]

    # Load the .npz file
    npzReader = NPZReader(left_features_path, right_features_path)

    # Iterate over each row in the extended_train_df
    for index, row in extended_train_df.iterrows():
        print("Processing row: ", index)
        start = time.time()
        left_image = row[0] + '.jpg'
        left_feature = npzReader.get_feature(left_image)

        for i in range(1, num_cols):  # Assuming there are 20 right images per row
            right_image = row[i] + '.jpg'
            right_feature = npzReader.get_feature(right_image)

            feature = np.concatenate((left_feature, right_feature), axis=0)
            # Compute cosine similarity
            probability = svm.predict_proba(feature)[1]

            # Replace the filename with the similarity value
            extended_train_df.iloc[index, i] = probability
        end = time.time()
        print(f"Time elapsed: {end - start}; ")

        # save the extended_train_df to a csv file
    extended_train_df.to_csv(output_path, index=False)

    return extended_train_df




if __name__ == '__main__':
    # extractor = VggFeatureExtractor()
    # extractor.save_features('../dataset/train/right', f'../dataset/train/right_features_{target_size[0]}_{target_size[1]}.npz')

    # feature = load_specific_feature('../dataset/train/left_features.npz', 'agp.jpg')
    # plot_feature(feature, title='agp.jpg')

    # compute_cosine_similarity_for_submission('../dataset/extended_train.csv',
    #                                          f'../dataset/train/left_features_{target_size[0]}_{target_size[1]}.npz',
    #                                          f'../dataset/train/right_features_{target_size[0]}_{target_size[1]}.npz',
    #                                          '../dataset/temp_test_submission.csv')
    # 0.4325

    features, labels = svm_for_submission('../dataset/extended_train.csv', '../dataset/train/left_features_224_224.npz', '../dataset/train/right_features_224_224.npz')
    svm_predict('../dataset/test_accuracy.csv', '../dataset/train/left_features_224_224.npz', '../dataset/train/right_features_224_224.npz', features, labels, '../dataset/temp_test_accuracy.csv')

