// Arquivo gerado automaticamente pelo Pipeline TinyML (TCC). Nao editar a mao.
#ifndef TEST_VECTORS_H_
#define TEST_VECTORS_H_

#include <stdint.h>

// Janelas do conjunto de TESTE (nunca vistas no treino), decimadas para
// 1000 Hz e armazenadas na flash para validacao embarcada.
constexpr int kNumTestVectors = 112;
constexpr int kTestVectorWindow = 512;
constexpr int kTestVectorClasses = 3;
constexpr int kTestVectorSampleRateHz = 1000;

// Reconstrucao do valor fisico: g = kTestVectorData[i] * kTestVectorScale.
constexpr float kTestVectorScale = 2.928110689e-06f;

// 112 x 512 amostras, em ordem [vetor][amostra].
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

#endif  // TEST_VECTORS_H_
