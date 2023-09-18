import numpy as np
import os

from keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

#TODO:
# 1. 试试 ResNet50, VGG19
# 2. 试试低级特征（从其他层捕获特征）

target_size = (224, 224)  # 包含全连接层fc1，所以size固定为224
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


if __name__ == '__main__':
    extractor = VggFeatureExtractor()
    extractor.save_features('../dataset/train/right', f'../dataset/train/right_features_{target_size[0]}_{target_size[1]}.npz')

    # feature = load_specific_feature('../dataset/train/left_features.npz', 'agp.jpg')
    # plot_feature(feature, title='agp.jpg')
