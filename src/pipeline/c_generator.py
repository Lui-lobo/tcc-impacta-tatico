import os
from typing import List

import numpy as np

_HEADER_NOTICE = "// Arquivo gerado automaticamente pelo Pipeline TinyML (TCC). Nao editar a mao."


def _c_array_body(values, per_line: int, formatter) -> str:
    """Formata uma sequencia numerica como corpo de inicializador C."""
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        lines.append("    " + ", ".join(formatter(v) for v in chunk))
    return ",\n".join(lines)


class CArtifactGenerator:
    @staticmethod
    def tflite_to_c_array(tflite_path: str, header_path: str, source_path: str,
                          array_name: str = "model_tflite") -> None:
        """Converte o .tflite em um par header/source C++.

        O array fica no .cpp (e nao no .h) para que o modelo possa ser incluido
        por varias unidades de compilacao sem gerar simbolos duplicados na
        linkagem, que e o padrao usado pelos exemplos do TFLite Micro.
        """
        with open(tflite_path, 'rb') as f:
            tflite_content = f.read()

        hex_lines = []
        for i in range(0, len(tflite_content), 12):
            chunk = tflite_content[i:i + 12]
            hex_lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk))
        hex_array_content = ",\n".join(hex_lines)

        guard = f"{array_name.upper()}_H_"
        header_code = f"""{_HEADER_NOTICE}
#ifndef {guard}
#define {guard}

// Modelo CNN 1D quantizado em int8, definido em {os.path.basename(source_path)}.
extern const unsigned char {array_name}[];
extern const unsigned int {array_name}_len;

#endif  // {guard}
"""

        # alignas(16) e exigido pelo flatbuffer do TFLite Micro para leitura direta da flash.
        source_code = f"""{_HEADER_NOTICE}
#include "{os.path.basename(header_path)}"

alignas(16) const unsigned char {array_name}[] = {{
{hex_array_content}
}};

const unsigned int {array_name}_len = {len(tflite_content)};
"""

        CArtifactGenerator._write(header_path, header_code)
        CArtifactGenerator._write(source_path, source_code)

    @staticmethod
    def generate_params_header(output_path: str, window_size: int, num_classes: int,
                               norm_mean: float, norm_std: float,
                               class_labels: List[str],
                               sample_rate_hz: int) -> None:
        """Exporta os parametros de pre-processamento usados no treino.

        Sem isso o ESP32 alimentaria o modelo com uma escala diferente da vista
        durante o treinamento, e a inferencia sairia sistematicamente errada.
        A taxa de amostragem viaja junto para que o firmware possa comparar a
        banda em que treinou com a banda que o sensor entrega.
        """
        labels = ", ".join(f'"{label}"' for label in class_labels)
        code = f"""{_HEADER_NOTICE}
#ifndef MODEL_PARAMS_H_
#define MODEL_PARAMS_H_

// Numero de amostras por janela de vibracao (deve bater com o tensor de entrada).
constexpr int kWindowSize = {window_size};
constexpr int kNumClasses = {num_classes};

// Taxa de amostragem das janelas de treino, apos a decimacao do dataset CWRU.
// O firmware deve amostrar o MPU6050 nesta mesma taxa.
constexpr int kTrainingSampleRateHz = {sample_rate_hz};

// Normalizacao estatistica aplicada no treino: (x - kNormMean) / kNormStd.
constexpr float kNormMean = {norm_mean:.10f}f;
constexpr float kNormStd = {norm_std:.10f}f;

// 'static' mantem uma copia por unidade de compilacao, evitando simbolos duplicados.
static const char* const kClassLabels[kNumClasses] = {{{labels}}};

#endif  // MODEL_PARAMS_H_
"""
        CArtifactGenerator._write(output_path, code)

    @staticmethod
    def generate_test_vectors(header_path: str, source_path: str,
                              windows_i16: np.ndarray, labels: np.ndarray,
                              reference_output: np.ndarray,
                              input_hash: np.ndarray, scale: float,
                              sample_rate_hz: int) -> None:
        """Grava o conjunto de teste do CWRU na flash do ESP32.

        Serve para validar o caminho de inferencia embarcado sem bancada
        rotativa: as mesmas janelas que produziram a acuracia do relatorio sao
        reprocessadas dentro do microcontrolador, e o resultado e comparado com
        a saida do interpretador TFLite de referencia (o do PC).

        As janelas viajam em int16 e nao em float32. Sao ~15 bits uteis contra
        os 8 bits que o quantizador de entrada preserva, entao a conversao nao
        perde nada que o modelo pudesse enxergar - e custa metade da flash.
        Alem disso, int16 e exatamente o formato que o MPU6050 entrega, o que
        mantem o vetor de teste no mesmo domicilio numerico do sinal real.

        `reference_output` guarda o tensor de saida int8 completo, e nao apenas
        a classe vencedora. Comparar os bytes permite afirmar identidade
        aritmetica entre PC e ESP32, uma alegacao bem mais forte do que
        concordancia de argmax.
        """
        n_vectors, window = windows_i16.shape
        n_classes = reference_output.shape[1]

        guard = "TEST_VECTORS_H_"
        header_code = f"""{_HEADER_NOTICE}
#ifndef {guard}
#define {guard}

#include <stdint.h>

// Janelas do conjunto de TESTE (nunca vistas no treino), decimadas para
// {sample_rate_hz} Hz e armazenadas na flash para validacao embarcada.
constexpr int kNumTestVectors = {n_vectors};
constexpr int kTestVectorWindow = {window};
constexpr int kTestVectorClasses = {n_classes};
constexpr int kTestVectorSampleRateHz = {sample_rate_hz};

// Reconstrucao do valor fisico: g = kTestVectorData[i] * kTestVectorScale.
constexpr float kTestVectorScale = {float(scale):.9e}f;

// {n_vectors} x {window} amostras, em ordem [vetor][amostra].
extern const int16_t kTestVectorData[];

// Classe verdadeira de cada janela (indice em kClassLabels).
extern const uint8_t kTestVectorLabel[];

// Tensor de saida int8 produzido pelo interpretador TFLite no PC, para as
// MESMAS janelas. Referencia de comparacao bit a bit.
extern const int8_t kTestVectorRefOutput[];

// Hash FNV-1a de 32 bits da janela JA quantizada em int8 no PC. Permite ao
// ESP32 provar que montou exatamente a mesma entrada, separando um erro de
// pre-processamento de uma diferenca entre os kernels do TFLite Micro e os do
// TFLite de desktop.
extern const uint32_t kTestVectorInputHash[];

#endif  // {guard}
"""

        data_body = _c_array_body(
            windows_i16.reshape(-1).tolist(), 16, lambda v: f"{v:6d}")
        label_body = _c_array_body(
            labels.reshape(-1).tolist(), 32, lambda v: f"{v}")
        ref_body = _c_array_body(
            reference_output.reshape(-1).tolist(), 24, lambda v: f"{v:4d}")
        hash_body = _c_array_body(
            input_hash.reshape(-1).tolist(), 6, lambda v: f"0x{v:08x}u")

        source_code = f"""{_HEADER_NOTICE}
#include "test_vectors.h"

const int16_t kTestVectorData[kNumTestVectors * kTestVectorWindow] = {{
{data_body}
}};

const uint8_t kTestVectorLabel[kNumTestVectors] = {{
{label_body}
}};

const int8_t kTestVectorRefOutput[kNumTestVectors * kTestVectorClasses] = {{
{ref_body}
}};

const uint32_t kTestVectorInputHash[kNumTestVectors] = {{
{hash_body}
}};
"""

        CArtifactGenerator._write(header_path, header_code)
        CArtifactGenerator._write(source_path, source_code)

    @staticmethod
    def _write(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
