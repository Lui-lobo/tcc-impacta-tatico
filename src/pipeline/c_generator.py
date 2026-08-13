import os
from typing import List

_HEADER_NOTICE = "// Arquivo gerado automaticamente pelo Pipeline TinyML (TCC). Nao editar a mao."


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
    def _write(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
