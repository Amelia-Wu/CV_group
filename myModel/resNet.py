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
        # 初始化一个MobileNetV2模型，并去掉顶部分类层
        base_model = MobileNetV2(weights='imagenet', include_top=False)

        # 添加自定义的分类层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        # 构建新的模型
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # 冻结预训练模型的权重，以防止在初始训练中过多更改
        for layer in base_model.layers:
            layer.trainable = False

        # 编译模型
        self.model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_data_generator, epochs):
        # 训练模型
        self.model.fit(train_data_generator, epochs=epochs)

    def predict(self, image_path):
        # 加载图像并进行预处理
        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # 进行预测
        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions)
        return predicted_class

    def save(self, path):
        # 保存模型到文件
        self.model.save(path)

    def load(self, path):
        # 加载模型文件
        self.model = tf.keras.models.load_model(path)
