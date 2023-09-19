import numpy as np


class NPZReader():
    def __init__(self, left_features_path, right_features_path):
        self.left_data = np.load(left_features_path)
        self.left_filenames = self.left_data['filenames']
        self.left_features = self.left_data['features']

        self.right_data = np.load(right_features_path)
        self.right_filenames = self.right_data['filenames']
        self.right_features = self.right_data['features']

    def get_feature(self, image_name):
        if image_name in self.left_filenames:
            ind = np.where(self.left_filenames == image_name)[0][0]
            feature = self.left_features[ind]
            return feature
        elif image_name in self.right_filenames:
            ind = np.where(self.right_filenames == image_name)[0][0]
            feature = self.right_features[ind]
            return feature
        else:
            print("Image not found in the npz file.")
            return None