import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
import numpy as np


class FineTunedModel:
    def __init__(self, num_classes):

        base_model = MobileNetV2(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)


        self.model = Model(inputs=base_model.input, outputs=predictions)


        for layer in base_model.layers:
            layer.trainable = False

        self.model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_data_generator, epochs):

        self.model.fit(train_data_generator, epochs=epochs)

    def predict(self, image_path):

        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)


        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions)
        return predicted_class

    def save(self, path):

        self.model.save(path)

    def load(self, path):

        self.model = tf.keras.models.load_model(path)
