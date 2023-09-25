import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

generated_data = pd.read_csv('./utils/new_data.csv')
class FineTunedModel:
    def __init__(self):
        # Define the ResNet50 base model (pre-trained and without top classification layers)
        resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # Add global average pooling layer
        x = GlobalAveragePooling2D()(resnet_model.output)
        # Add a dense layer with sigmoid activation for binary classification
        output = Dense(1, activation='sigmoid')(x)
        # Create a new model with the ResNet50 backbone and the custom output layer
        self.model = Model(inputs=resnet_model.input, outputs=output)

    def extract(self, img_path):
        # Load image with size (224, 224)
        img = image.load_img(img_path, target_size=(224, 224))
        # Convert image to array
        img_array = image.img_to_array(img)
        # Convert to a batch of size (1, 224, 224, 3)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        # Preprocess the input for Resnet model
        preprocessed_img = preprocess_input(expanded_img_array)
        # Get features
        features = self.model.predict(preprocessed_img)
        return features

    def combine_features(self, left_img_path, right_img_path):
        left_features = self.extract(left_img_path)
        right_features = self.extract(right_img_path)
        # Add left features and right features together
        combined_features = left_features + right_features
        # self.combine_features_list.append(combined_features)
        return combined_features

    def get_feature_label_data(self, dataset):
        combined_feature_list = []
        for index, row in dataset.iterrows():
            left_img, right_img, label = row['left'], row['right'], row['label']
            left_img_path = './dataset/train/left/' + left_img + '.jpg'
            right_img_path = './dataset/train/right/' + right_img + '.jpg'
            combined_feature = self.combine_features(left_img_path, right_img_path)
            print(combined_feature)
            combined_feature_list.append(combined_feature)
        return combined_feature_list

    def split_data(self):
        all_data = generated_data.iloc[1:]
        all_features = np.array(self.get_feature_label_data(all_data))
        all_labels = np.array(all_data.iloc[:, -1:].to_numpy())

        # Create an array of indices
        num_samples = len(all_data)  # Replace 'data' with your dataset
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Determine the split size
        split_ratio = 0.7  # for a 70-30 split
        split_index = int(split_ratio * num_samples)

        # Split the indices
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        # Create the training and testing sets based on the selected indices
        x_train = all_features[train_indices]
        y_train = all_labels[train_indices]
        x_test = all_features[test_indices]
        y_test = all_labels[test_indices]

        # x_train, x_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3)
        return x_train, x_test, y_train, y_test

    def train(self, x_train, y_train):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        resnet_training = self.model.fit(x=np.asarray(x_train),
                                         y=np.asarray(y_train),
                                         epochs=10,
                                         batch_size=100,
                                         verbose=1)
        # Plot the accuracy of train dataset and validation dataset
        plt.plot(resnet_training.history['accuracy'], label='Train')
        # plt.plot(resnet_training.history['val_accuracy'], label='Validation')
        plt.xticks(np.arange(0, 20, step=2))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train accuracy and validation accuracy')
        plt.legend()
        plt.show()

    def evaluate(self, x_test, y_test):
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print(f'Loss: {loss}, Accuracy: {accuracy}')

    def predict(self, image_path):

        img = image.load_img(image_path, target_size=(224,224))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        predictions = self.model.predict(img)
        probability_of_class_1 = predictions[0][0]
        return probability_of_class_1

    def save(self, path):

        self.model.save(path)

    def load(self, path):

        self.model = tf.keras.models.load_model(path)
