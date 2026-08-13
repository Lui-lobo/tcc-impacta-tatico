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

    def _static_batch_model(self) -> tf.keras.Model:
        """Reembrulha o modelo treinado com o batch fixo em 1.

        Com o batch dinamico (None), o Flatten do Keras 3 vira uma sequencia
        SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE: o grafo calcula a forma de
        saida em tempo de execucao. No TensorFlow Lite Micro isso obriga a
        registrar tres operadores a mais e mantem um subgrafo dinamico dentro
        de um interpretador que so trabalha com memoria estatica.

        Fixando o batch em 1 - que e exatamente como o ESP32 infere, uma janela
        por vez - a forma passa a ser conhecida em tempo de conversao e o
        TFLite dobra as quatro operacoes numa unica constante.

        O wrapper reutiliza as mesmas camadas, entao os pesos treinados sao
        compartilhados, nao copiados.
        """
        batch_shape = (1,) + tuple(self.model.input_shape[1:])
        inputs = tf.keras.Input(batch_shape=batch_shape)
        return tf.keras.Model(inputs, self.model(inputs))

    def quantize_to_int8(self, output_path: str) -> None:
        converter = tf.lite.TFLiteConverter.from_keras_model(self._static_batch_model())
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self._representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_quant_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_quant_model)
