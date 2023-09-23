from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
from utils.preprocess_data import get_data_pairs

from tensorflow.keras.applications import VGG16


def get_model(num_classes=2):
    vgg16_base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    vgg16_base.trainable = True

    model = keras.Sequential([
        vgg16_base,
        keras.layers.Flatten(),  # 将VGG16的输出展平
        keras.layers.Dense(4096, activation='relu'),  # VGG16原始的全连接层
        keras.layers.Dense(4096, activation='relu'),  # VGG16原始的全连接层
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

if __name__=="__main__":
    train_dataset, val_dataset = get_data_pairs(
        'E:/3Melbourne_uni_2023_S2/cv/assignment/Group/code/utils/dataset_generated.csv')
    num_train = tf.data.experimental.cardinality(train_dataset)
    num_val = tf.data.experimental.cardinality(val_dataset)
    print(f"Number of training examples: {num_train}")
    print(f"Number of validation examples: {num_val}")
    # batch_size = 32, so 32*1250 = 40000(2000 left images * 20 right images)
    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=8)
