// Arquivo gerado automaticamente pelo Pipeline TinyML (TCC). Nao editar a mao.
#ifndef MODEL_PARAMS_H_
#define MODEL_PARAMS_H_

// Numero de amostras por janela de vibracao (deve bater com o tensor de entrada).
constexpr int kWindowSize = 512;
constexpr int kNumClasses = 3;

// Taxa de amostragem das janelas de treino, apos a decimacao do dataset CWRU.
// O firmware deve amostrar o MPU6050 nesta mesma taxa.
constexpr int kTrainingSampleRateHz = 1000;

// Normalizacao estatistica aplicada no treino: (x - kNormMean) / kNormStd.
constexpr float kNormMean = 0.0080134990f;
constexpr float kNormStd = 0.0345664136f;

// 'static' mantem uma copia por unidade de compilacao, evitando simbolos duplicados.
static const char* const kClassLabels[kNumClasses] = {"0_normal", "1_inner_race", "2_outer_race"};

#endif  // MODEL_PARAMS_H_
