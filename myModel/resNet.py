import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

generated_data = pd.read_csv('./utils/dataset_generated.csv')
test_candidates_data = pd.read_csv('./dataset/test_candidates.csv')

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

    def process_img(self, img_path):
        # Load image with size (224, 224)
        img = image.load_img(img_path, target_size=(224, 224))
        # Convert image to array
        img_array = image.img_to_array(img)
        # Preprocess the input for Resnet model
        preprocessed_img = preprocess_input(img_array)
        return preprocessed_img

    def combine_img(self, left_img_path, right_img_path):
        left_img = self.process_img(left_img_path)
        right_img = self.process_img(right_img_path)
        # Add left features and right features together
        combined_imgs = left_img + right_img
        # print(combined_imgs)
        return combined_imgs

    def get_combined_img_list(self, dataset):
        combined_img_list = []
        for index, row in dataset.iterrows():
            left_img, right_img, label = row['left'], row['right'], row['label']
            left_img_path = './dataset/train/left/' + left_img + '.jpg'
            right_img_path = './dataset/train/right/' + right_img + '.jpg'
            combined_img = self.combine_img(left_img_path, right_img_path)
            combined_img_list.append(combined_img)
        return combined_img_list

    def split_data(self):
        all_data = generated_data.iloc[1:]
        all_imgs = np.array(self.get_combined_img_list(all_data))
        all_labels = np.array(all_data.iloc[:, -1].to_numpy())

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
        x_train = all_imgs[train_indices]
        y_train = all_labels[train_indices]
        x_test = all_imgs[test_indices]
        y_test = all_labels[test_indices]

        # x_train, x_test, y_train, y_test = train_test_split(all_imgs, all_labels, test_size=0.3)
        return x_train, x_test, y_train, y_test

    def train(self, x_train, y_train):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        resnet_training = self.model.fit(x=np.asarray(x_train),
                                         y=np.asarray(y_train),
                                         epochs=20,
                                         batch_size=100,
                                         verbose=1)

        # Plot the accuracy of train dataset and validation dataset
        plt.plot(resnet_training.history['accuracy'], label='Train')
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

    def predict(self, dataset, output_path):
        num_cols = dataset.shape[1]
        for index, row in dataset.iterrows():
            print("Processing row: ", index)
            start = time.time()
            left_img = row['left']
            left_img_path = './dataset/test/left/' + left_img + '.jpg'

            for i in range(1, num_cols):
                right_img = row[i]
                right_img_path = './dataset/test/right/' + right_img + '.jpg'
                combined_img = self.combine_img(left_img_path, right_img_path)

                img = image.img_to_array(combined_img)
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                predictions = self.model.predict(img)
                probability_of_class_1 = predictions[0][0]

                # Replace the filename with the similarity value
                dataset.iloc[index, i] = probability_of_class_1

            end = time.time()
            print(f"Time elapsed: {end - start}; ")
        # save the extended_train_df to a csv file
        dataset.to_csv(output_path, index=False)

        return dataset

    def save(self, path):

        self.model.save(path)

    def load(self, path):

        self.model = tf.keras.models.load_model(path)
