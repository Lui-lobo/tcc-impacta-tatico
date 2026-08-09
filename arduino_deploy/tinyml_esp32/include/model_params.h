// Arquivo gerado automaticamente pelo Pipeline TinyML (TCC). Nao editar a mao.
#ifndef MODEL_PARAMS_H_
#define MODEL_PARAMS_H_

// Numero de amostras por janela de vibracao (deve bater com o tensor de entrada).
constexpr int kWindowSize = 512;
constexpr int kNumClasses = 3;

// Normalizacao estatistica aplicada no treino: (x - kNormMean) / kNormStd.
constexpr float kNormMean = 0.0154723665f;
constexpr float kNormStd = 0.3682984114f;

// 'static' mantem uma copia por unidade de compilacao, evitando simbolos duplicados.
static const char* const kClassLabels[kNumClasses] = {"0_normal", "1_inner_race", "2_outer_race"};

#endif  // MODEL_PARAMS_H_
