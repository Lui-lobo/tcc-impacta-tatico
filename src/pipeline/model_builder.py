import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

class TinyMLModelBuilder:
    @staticmethod
    def build_cnn1d(input_shape: Tuple[int, int], num_classes: int) -> models.Sequential:
        model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Conv1D(filters=8, kernel_size=16, strides=2, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=16, kernel_size=8, strides=2, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
