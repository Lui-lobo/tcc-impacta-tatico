import tensorflow as tf
import numpy as np

class ModelQuantizer:
    def __init__(self, keras_model: tf.keras.Model, representative_dataset: np.ndarray):
        self.model = keras_model
        self.rep_data = representative_dataset

    def _representative_data_gen(self):
        indices = np.random.choice(len(self.rep_data), size=min(150, len(self.rep_data)), replace=False)
        for i in indices:
            yield [self.rep_data[i:i+1].astype(np.float32)]

    def quantize_to_int8(self, output_path: str) -> None:
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self._representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_quant_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_quant_model)
