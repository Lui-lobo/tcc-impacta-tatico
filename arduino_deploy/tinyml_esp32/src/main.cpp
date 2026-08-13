/*
 * Deploy TinyML no ESP32 - TCC Manutencao Preditiva de Rolamentos
 *
 * Classifica janelas de 512 amostras de vibracao em: normal, falha na pista
 * interna ou falha na pista externa, usando a CNN 1D quantizada em int8
 * gerada pelo pipeline Python (build.py).
 *
 * ---------------------------------------------------------------------------
 * VERSAO INSTRUMENTADA (v2) - foco em diagnosticar a aquisicao do sinal
 * ---------------------------------------------------------------------------
 * A v1 lia o MPU6050 em loop aberto, na velocidade maxima do barramento I2C, e
 * alimentava o modelo com o eixo Z bruto. Isso produzia dois defeitos graves:
 *
 *   (a) DC da gravidade: o eixo Z em repouso vale ~1.0 g constante. O dataset
 *       CWRU e um sinal AC de media ~0 (kNormMean = 0.015). O vetor de entrada
 *       virava uma constante ~2.67 apos a normalizacao, e a saida do modelo
 *       ficava congelada no mesmo valor a cada iteracao.
 *   (b) Sobreamostragem: o MPU6050 atualiza os registradores do acelerometro a
 *       no maximo 1 kHz. Ler mais rapido que isso apenas repete a amostra
 *       anterior, destruindo a forma de onda.
 *
 * Esta versao corrige os dois pontos e instrumenta cada etapa do caminho
 * sensor -> I2C -> janela -> normalizacao -> quantizacao -> inferencia, para
 * que o Serial Monitor mostre exatamente onde o sinal se perde.
 */
#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_params.h"
#include "model_tflite.h"

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ===========================================================================
// Configuracao de aquisicao
// ===========================================================================

// Pinos do barramento I2C.
constexpr int kPinSda = 21;
constexpr int kPinScl = 22;
constexpr uint32_t kI2cClockHz = 400000;

// Taxa de amostragem alvo. Vem de model_params.h, gerado pelo build.py, para
// que firmware e treino nunca divirjam. O acelerometro do MPU6050 tem taxa
// interna de atualizacao de 1 kHz; pedir mais que isso so devolve repeticoes.
constexpr uint32_t kDefaultSampleRateHz = (uint32_t)kTrainingSampleRateHz;

// Fundo de escala do acelerometro. +-2 g da a melhor resolucao (16384 LSB/g);
// se o log acusar amostras saturadas, suba para 4 g ou 8 g.
constexpr mpu6050_accel_range_t kAccelRange = MPU6050_RANGE_2_G;

// Filtro passa-baixa interno. BAND_260_HZ = DLPF desligado (banda de 260 Hz no
// acelerometro, taxa base de 8 kHz para o divisor de amostragem).
constexpr mpu6050_bandwidth_t kFilterBandwidth = MPU6050_BAND_260_HZ;

// Eixo entregue ao modelo.
enum AxisMode : uint8_t { AXIS_X = 0, AXIS_Y = 1, AXIS_Z = 2, AXIS_MAGNITUDE = 3 };
constexpr AxisMode kDefaultAxis = AXIS_Z;

// Remove o valor medio da janela antes de normalizar. Indispensavel: elimina a
// componente da gravidade e deixa apenas a vibracao, como no dataset CWRU.
constexpr bool kDefaultRemoveDc = true;

// Abaixo deste desvio padrao (em g) a janela e considerada praticamente
// estatica e o firmware emite um alerta em vez de fingir que classificou algo.
constexpr float kStaticSignalThresholdG = 0.002f;

// Intervalo entre janelas (ms).
constexpr uint32_t kLoopDelayMs = 2000;

// ===========================================================================
// Registradores do MPU6050 (acesso direto, sem passar pela Adafruit)
// ===========================================================================
// A leitura da janela usa Wire cru em vez de mpu.getEvent() por dois motivos:
// getEvent() transfere 14 bytes (accel + temp + giro) e faz conversao para
// float dentro do laco critico. Ler so os 6 bytes do acelerometro reduz o
// tempo de barramento em ~55% e torna o intervalo entre amostras estavel.

constexpr uint8_t kMpuI2cAddr = 0x68;
constexpr uint8_t REG_SMPLRT_DIV = 0x19;
constexpr uint8_t REG_CONFIG = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_INT_ENABLE = 0x38;
constexpr uint8_t REG_INT_STATUS = 0x3A;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_WHO_AM_I = 0x75;

// ===========================================================================
// Estado global
// ===========================================================================

Adafruit_MPU6050 mpu;

// Area de trabalho do interpretador (tensores intermediarios + metadados).
// O valor real consumido e impresso no setup(); ajuste se sobrar/faltar memoria.
constexpr int kTensorArenaSize = 24 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input = nullptr;
static TfLiteTensor* model_output = nullptr;

// Janela crua do sensor: 3 eixos em LSB, exatamente como saiu do registrador.
static int16_t raw_window[kWindowSize][3];

// Janela em unidades de 'g', ja com o eixo escolhido e o DC removido.
static float accel_buffer[kWindowSize];

// Parametros ajustaveis em tempo de execucao pelo console serial.
static uint32_t g_sample_rate_hz = kDefaultSampleRateHz;
static uint32_t g_sample_interval_us = 1000000UL / kDefaultSampleRateHz;
static AxisMode g_axis = kDefaultAxis;
static bool g_remove_dc = kDefaultRemoveDc;
static bool g_verbose = false;
static bool g_paused = false;
static uint32_t g_window_count = 0;

// LSB por g conforme o fundo de escala configurado.
static float g_lsb_per_g = 16384.0f;

// Metricas de uma aquisicao de janela.
struct SamplingReport {
  uint32_t interval_us;      // intervalo teorico entre amostras
  uint32_t duration_us;      // duracao real da janela inteira
  float actual_fs_hz;        // taxa efetivamente atingida
  uint32_t max_jitter_us;    // maior desvio em relacao ao instante teorico
  uint32_t late_samples;     // amostras que perderam o prazo (fs alta demais)
  uint32_t i2c_errors;       // falhas de transacao no barramento
  uint32_t repeated_samples; // leituras identicas a anterior (sobreamostragem)
  uint32_t saturated_samples;// amostras no fundo de escala (clipping)
};

// Estatisticas de um eixo dentro da janela.
struct AxisStats {
  float mean;  // media (contem a gravidade)
  float std;   // desvio padrao (energia da vibracao)
  float min;
  float max;
};

// ===========================================================================
// Acesso de baixo nivel ao sensor
// ===========================================================================

static bool mpuReadRegs(uint8_t reg, uint8_t* dst, uint8_t len) {
  Wire.beginTransmission(kMpuI2cAddr);
  Wire.write(reg);
  // endTransmission(false) mantem o barramento com repeated-start, que e o
  // protocolo esperado pelo MPU6050 para leitura de registrador.
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  if (Wire.requestFrom((uint16_t)kMpuI2cAddr, len) != len) {
    return false;
  }
  for (uint8_t i = 0; i < len; i++) {
    dst[i] = Wire.read();
  }
  return true;
}

static bool mpuWriteReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(kMpuI2cAddr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission(true) == 0;
}

// Leitura dos 6 bytes do acelerometro. Os shifts ficam em variaveis separadas
// porque a ordem de avaliacao dos operandos de '|' nao e garantida em C++.
static inline bool readAccelRaw(int16_t* out_xyz) {
  uint8_t buf[6];
  if (!mpuReadRegs(REG_ACCEL_XOUT_H, buf, 6)) {
    return false;
  }
  out_xyz[0] = (int16_t)(((uint16_t)buf[0] << 8) | buf[1]);
  out_xyz[1] = (int16_t)(((uint16_t)buf[2] << 8) | buf[3]);
  out_xyz[2] = (int16_t)(((uint16_t)buf[4] << 8) | buf[5]);
  return true;
}

// Espera ativa ate um instante absoluto. Cede a CPU quando a espera e longa,
// para nao inanir o idle task (watchdog) em taxas de amostragem baixas.
static inline void waitUntilUs(uint32_t target_us) {
  for (;;) {
    const int32_t remaining = (int32_t)(target_us - micros());
    if (remaining <= 0) {
      return;
    }
    if (remaining > 2000) {
      delay((uint32_t)(remaining - 1000) / 1000);
    }
  }
}

// ===========================================================================
// Diagnostico do barramento e do sensor
// ===========================================================================

static void scanI2cBus() {
  Serial.println("[I2C] Varredura do barramento:");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("      dispositivo em 0x%02X%s\n", addr,
                    (addr == kMpuI2cAddr) ? "  <- MPU6050" : "");
      found++;
    }
  }
  if (found == 0) {
    Serial.println("      NENHUM dispositivo encontrado. Verifique SDA/SCL, "
                   "alimentacao 3V3 e os resistores de pull-up.");
  }
}

static void dumpMpuConfig() {
  struct {
    const char* name;
    uint8_t reg;
  } regs[] = {
      {"WHO_AM_I    ", REG_WHO_AM_I},   {"PWR_MGMT_1  ", REG_PWR_MGMT_1},
      {"SMPLRT_DIV  ", REG_SMPLRT_DIV}, {"CONFIG(DLPF)", REG_CONFIG},
      {"GYRO_CONFIG ", REG_GYRO_CONFIG}, {"ACCEL_CONFIG", REG_ACCEL_CONFIG},
      {"INT_ENABLE  ", REG_INT_ENABLE},
  };
  Serial.println("[MPU] Registradores de configuracao:");
  for (auto& r : regs) {
    uint8_t v = 0;
    if (mpuReadRegs(r.reg, &v, 1)) {
      Serial.printf("      %s (0x%02X) = 0x%02X\n", r.name, r.reg, v);
    } else {
      Serial.printf("      %s (0x%02X) = ERRO DE LEITURA\n", r.name, r.reg);
    }
  }
}

// Mede quanto tempo custa, na pratica, uma leitura de 6 bytes do acelerometro.
// Define o teto real de taxa de amostragem atingivel neste hardware.
static uint32_t benchmarkAccelRead(int iterations) {
  int16_t xyz[3];
  const uint32_t t0 = micros();
  for (int i = 0; i < iterations; i++) {
    readAccelRaw(xyz);
  }
  return (micros() - t0) / (uint32_t)iterations;
}

// Mede a taxa de saida REAL do sensor contando quantas vezes o bit DATA_RDY
// sobe. Responde diretamente a pergunta "o sensor esta produzindo amostras
// novas na taxa que eu configurei?" - independentemente do que o ESP32 pede.
static uint32_t measureSensorOdrHz(uint32_t duration_ms) {
  uint8_t status = 0;
  mpuReadRegs(REG_INT_STATUS, &status, 1);  // limpa pendencia
  uint32_t new_samples = 0;
  const uint32_t t0 = millis();
  while ((millis() - t0) < duration_ms) {
    if (mpuReadRegs(REG_INT_STATUS, &status, 1) && (status & 0x01)) {
      new_samples++;
    }
  }
  return (new_samples * 1000UL) / duration_ms;
}

// Aplica a taxa de amostragem pedida ao divisor interno do sensor.
// Sample Rate = Gyro Output Rate / (1 + SMPLRT_DIV), onde a taxa base e de
// 8 kHz com o DLPF desligado e 1 kHz com o DLPF ativo.
static void applySampleRate(uint32_t fs_hz) {
  if (fs_hz == 0) {
    fs_hz = 1;
  }
  g_sample_rate_hz = fs_hz;
  g_sample_interval_us = 1000000UL / fs_hz;

  const uint32_t base_hz = (kFilterBandwidth == MPU6050_BAND_260_HZ) ? 8000UL : 1000UL;
  uint32_t divisor = (fs_hz >= base_hz) ? 0 : (base_hz / fs_hz) - 1;
  if (divisor > 255) {
    divisor = 255;
  }
  mpu.setSampleRateDivisor((uint8_t)divisor);

  Serial.printf("[MPU] fs alvo = %lu Hz | base = %lu Hz | SMPLRT_DIV = %lu "
                "| intervalo = %lu us\n",
                (unsigned long)fs_hz, (unsigned long)base_hz,
                (unsigned long)divisor, (unsigned long)g_sample_interval_us);

  if (fs_hz > 1000) {
    Serial.println("      [ALERTA] O acelerometro do MPU6050 atualiza a 1 kHz. "
                   "Acima disso as amostras se repetem.");
  }
}

static void runSensorDiagnostics() {
  Serial.println();
  Serial.println("----- DIAGNOSTICO DO SENSOR -----");
  dumpMpuConfig();

  const uint32_t read_us = benchmarkAccelRead(200);
  Serial.printf("[MPU] Custo medio de uma leitura de 6 bytes: %lu us "
                "(=> teto de ~%lu Hz de amostragem)\n",
                (unsigned long)read_us,
                (unsigned long)(read_us ? 1000000UL / read_us : 0));

  const uint32_t odr = measureSensorOdrHz(200);
  Serial.printf("[MPU] Taxa de saida REAL medida via DATA_RDY: %lu Hz "
                "(configurada: %lu Hz)\n",
                (unsigned long)odr, (unsigned long)g_sample_rate_hz);
  if (odr + odr / 10 < g_sample_rate_hz) {
    Serial.println("      [ALERTA] O sensor entrega menos amostras novas do "
                   "que a taxa pedida. A janela tera valores repetidos.");
  }

  // Verifica se os registradores realmente mudam entre leituras espacadas.
  int16_t a[3], b[3];
  readAccelRaw(a);
  delayMicroseconds(g_sample_interval_us);
  readAccelRaw(b);
  Serial.printf("[MPU] Duas leituras espacadas de %lu us: "
                "(%d,%d,%d) -> (%d,%d,%d) %s\n",
                (unsigned long)g_sample_interval_us, a[0], a[1], a[2], b[0],
                b[1], b[2],
                (a[0] == b[0] && a[1] == b[1] && a[2] == b[2])
                    ? "[IDENTICAS - suspeito]"
                    : "[OK - variando]");
  Serial.println("---------------------------------");
  Serial.println();
}

// ===========================================================================
// Aquisicao da janela
// ===========================================================================

// Preenche raw_window respeitando um cronograma ABSOLUTO de amostragem: cada
// amostra i deve sair em t0 + i*intervalo. Usar o instante absoluto (em vez de
// somar o intervalo a cada iteracao) impede o acumulo de atraso ao longo das
// 512 amostras.
static void sampleWindow(SamplingReport& rep) {
  rep = SamplingReport{};
  rep.interval_us = g_sample_interval_us;

  const int16_t saturation = 32767;
  int16_t prev[3] = {0, 0, 0};

  // Sincroniza o inicio da janela com uma amostra nova do sensor, para que a
  // primeira leitura nao seja um valor ja envelhecido no registrador.
  uint8_t status = 0;
  const uint32_t sync_start = millis();
  while ((millis() - sync_start) < 5) {
    if (mpuReadRegs(REG_INT_STATUS, &status, 1) && (status & 0x01)) {
      break;
    }
  }

  const uint32_t t0 = micros();
  for (int i = 0; i < kWindowSize; i++) {
    const uint32_t target = t0 + (uint32_t)i * g_sample_interval_us;
    const int32_t slack = (int32_t)(target - micros());
    if (slack > 0) {
      waitUntilUs(target);
    } else if (i > 0) {
      // Ja passamos do instante teorico: a taxa pedida e alta demais para o
      // custo da leitura I2C. Contabiliza e segue no ritmo maximo possivel.
      rep.late_samples++;
    }

    const uint32_t actual = micros();
    const uint32_t deviation =
        (actual > target) ? (actual - target) : (target - actual);
    if (deviation > rep.max_jitter_us) {
      rep.max_jitter_us = deviation;
    }

    int16_t xyz[3];
    if (!readAccelRaw(xyz)) {
      rep.i2c_errors++;
      // Repete a ultima amostra valida para nao inserir um degrau artificial.
      xyz[0] = prev[0];
      xyz[1] = prev[1];
      xyz[2] = prev[2];
    }

    if (i > 0 && xyz[0] == prev[0] && xyz[1] == prev[1] && xyz[2] == prev[2]) {
      rep.repeated_samples++;
    }
    for (int k = 0; k < 3; k++) {
      if (xyz[k] >= saturation || xyz[k] <= -saturation) {
        rep.saturated_samples++;
        break;
      }
    }
    prev[0] = xyz[0];
    prev[1] = xyz[1];
    prev[2] = xyz[2];

    raw_window[i][0] = xyz[0];
    raw_window[i][1] = xyz[1];
    raw_window[i][2] = xyz[2];
  }

  rep.duration_us = micros() - t0;
  rep.actual_fs_hz =
      (rep.duration_us > 0)
          ? (float)(kWindowSize - 1) * 1000000.0f / (float)rep.duration_us
          : 0.0f;
}

static AxisStats computeAxisStats(int axis) {
  AxisStats s;
  double sum = 0.0;
  s.min = 1e9f;
  s.max = -1e9f;
  for (int i = 0; i < kWindowSize; i++) {
    const float v = (float)raw_window[i][axis] / g_lsb_per_g;
    sum += v;
    if (v < s.min) s.min = v;
    if (v > s.max) s.max = v;
  }
  s.mean = (float)(sum / kWindowSize);

  double var = 0.0;
  for (int i = 0; i < kWindowSize; i++) {
    const float d = (float)raw_window[i][axis] / g_lsb_per_g - s.mean;
    var += (double)d * d;
  }
  s.std = sqrtf((float)(var / kWindowSize));
  return s;
}

// Converte a janela crua no vetor float que sera normalizado e quantizado.
// Retorna o desvio padrao do sinal resultante (em g).
static float buildInputSignal() {
  double sum = 0.0;
  for (int i = 0; i < kWindowSize; i++) {
    float v;
    if (g_axis == AXIS_MAGNITUDE) {
      const float x = (float)raw_window[i][0] / g_lsb_per_g;
      const float y = (float)raw_window[i][1] / g_lsb_per_g;
      const float z = (float)raw_window[i][2] / g_lsb_per_g;
      v = sqrtf(x * x + y * y + z * z);
    } else {
      v = (float)raw_window[i][(int)g_axis] / g_lsb_per_g;
    }
    accel_buffer[i] = v;
    sum += v;
  }

  // Remocao do DC: tira a gravidade e o offset de fabrica, deixando so a
  // componente AC de vibracao - que e o que o dataset CWRU contem.
  const float dc = (float)(sum / kWindowSize);
  if (g_remove_dc) {
    for (int i = 0; i < kWindowSize; i++) {
      accel_buffer[i] -= dc;
    }
  }

  double var = 0.0;
  for (int i = 0; i < kWindowSize; i++) {
    const float d = accel_buffer[i] - (g_remove_dc ? 0.0f : dc);
    var += (double)d * d;
  }
  return sqrtf((float)(var / kWindowSize));
}

// ===========================================================================
// Console serial
// ===========================================================================

static const char* axisName(AxisMode a) {
  switch (a) {
    case AXIS_X: return "X";
    case AXIS_Y: return "Y";
    case AXIS_Z: return "Z";
    default: return "|XYZ|";
  }
}

static void printHelp() {
  Serial.println();
  Serial.println("----- COMANDOS DO CONSOLE -----");
  Serial.println("  h  ajuda");
  Serial.println("  s  reexecuta o diagnostico do sensor");
  Serial.println("  d  liga/desliga o modo verboso (amostras cruas)");
  Serial.println("  x/y/z/m  seleciona o eixo enviado ao modelo (m = modulo)");
  Serial.println("  o  liga/desliga a remocao do nivel DC (gravidade)");
  Serial.println("  f  alterna fs: 250 -> 500 -> 1000 -> 2000 Hz");
  Serial.println("  r  despeja a janela atual em CSV (para FFT no Python)");
  Serial.println("  p  pausa/retoma a inferencia");
  Serial.println("-------------------------------");
  Serial.println();
}

static void dumpWindowCsv() {
  Serial.println();
  Serial.printf("# CSV fs=%lu Hz eixo=%s dc_removido=%d janela=%lu\n",
                (unsigned long)g_sample_rate_hz, axisName(g_axis),
                g_remove_dc ? 1 : 0, (unsigned long)g_window_count);
  Serial.println("# i,ax_g,ay_g,az_g,entrada_modelo");
  for (int i = 0; i < kWindowSize; i++) {
    Serial.printf("%d,%.6f,%.6f,%.6f,%.6f\n", i,
                  (float)raw_window[i][0] / g_lsb_per_g,
                  (float)raw_window[i][1] / g_lsb_per_g,
                  (float)raw_window[i][2] / g_lsb_per_g, accel_buffer[i]);
  }
  Serial.println("# fim");
  Serial.println();
}

static void handleSerialCommand() {
  while (Serial.available() > 0) {
    const int c = Serial.read();
    switch (c) {
      case 'h': printHelp(); break;
      case 's': runSensorDiagnostics(); break;
      case 'd':
        g_verbose = !g_verbose;
        Serial.printf("[CFG] modo verboso: %s\n", g_verbose ? "ON" : "OFF");
        break;
      case 'x': case 'y': case 'z': case 'm':
        g_axis = (c == 'x') ? AXIS_X
               : (c == 'y') ? AXIS_Y
               : (c == 'z') ? AXIS_Z
                            : AXIS_MAGNITUDE;
        Serial.printf("[CFG] eixo de entrada: %s\n", axisName(g_axis));
        break;
      case 'o':
        g_remove_dc = !g_remove_dc;
        Serial.printf("[CFG] remocao de DC: %s\n", g_remove_dc ? "ON" : "OFF");
        break;
      case 'f': {
        const uint32_t next = (g_sample_rate_hz == 250)    ? 500
                              : (g_sample_rate_hz == 500)  ? 1000
                              : (g_sample_rate_hz == 1000) ? 2000
                                                           : 250;
        applySampleRate(next);
        break;
      }
      case 'r': dumpWindowCsv(); break;
      case 'p':
        g_paused = !g_paused;
        Serial.printf("[CFG] inferencia: %s\n", g_paused ? "PAUSADA" : "ATIVA");
        break;
      default: break;
    }
  }
}

// ===========================================================================
// Setup
// ===========================================================================

static void haltOnError(const char* message) {
  Serial.println(message);
  while (true) {
    delay(1000);
  }
}

static void setupSensor() {
  Wire.begin(kPinSda, kPinScl);
  Wire.setClock(kI2cClockHz);
  Serial.printf("[I2C] SDA=%d SCL=%d clock=%lu Hz\n", kPinSda, kPinScl,
                (unsigned long)kI2cClockHz);

  scanI2cBus();

  uint8_t who = 0;
  if (!mpuReadRegs(REG_WHO_AM_I, &who, 1)) {
    haltOnError("ERRO: nao foi possivel ler WHO_AM_I. Barramento I2C mudo.");
  }
  Serial.printf("[MPU] WHO_AM_I = 0x%02X (esperado 0x68)\n", who);

  if (!mpu.begin(kMpuI2cAddr, &Wire)) {
    haltOnError("ERRO: falha ao inicializar o MPU6050. Verifique as conexoes.");
  }

  mpu.setAccelerometerRange(kAccelRange);
  mpu.setFilterBandwidth(kFilterBandwidth);

  // Habilita o bit DATA_RDY em INT_STATUS. Nao precisa do pino INT ligado:
  // o bit e lido por polling e serve para medir a taxa real do sensor.
  mpuWriteReg(REG_INT_ENABLE, 0x01);

  switch (kAccelRange) {
    case MPU6050_RANGE_2_G:  g_lsb_per_g = 16384.0f; break;
    case MPU6050_RANGE_4_G:  g_lsb_per_g = 8192.0f;  break;
    case MPU6050_RANGE_8_G:  g_lsb_per_g = 4096.0f;  break;
    default:                 g_lsb_per_g = 2048.0f;  break;
  }
  Serial.printf("[MPU] Fundo de escala: +-%d g (%.0f LSB/g) | banda do filtro: "
                "indice %d\n",
                2 << (int)kAccelRange, g_lsb_per_g, (int)kFilterBandwidth);

  applySampleRate(kDefaultSampleRateHz);
  runSensorDiagnostics();
}

static void setupModel() {
  Serial.printf("[TFL] Modelo na flash: %u bytes\n", model_tflite_len);

  const tflite::Model* model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    haltOnError("ERRO: schema do modelo incompativel com a versao da biblioteca.");
  }

  // Resolver enxuto em vez do AllOpsResolver, para economizar flash.
  // O modelo atual usa 5 operadores: RESHAPE, CONV_2D, MAX_POOL_2D,
  // FULLY_CONNECTED e SOFTMAX. EXPAND_DIMS fica registrado por precaucao: o
  // rebaixamento de Conv1D para Conv2D o reintroduz em algumas versoes do
  // conversor, e o custo de manter um operador a mais e desprezivel.
  // Se aparecer "Didn't find op for builtin opcode" no boot, acrescente o
  // operador aqui e aumente o parametro de template.
  static tflite::MicroMutableOpResolver<6> resolver;
  if (resolver.AddExpandDims() != kTfLiteOk ||
      resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddReshape() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) {
    haltOnError("ERRO: falha ao registrar os operadores do modelo.");
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    // Duas causas distintas caem aqui. Se as linhas acima trazem
    // "Didn't find op for builtin opcode 'XXX'", o modelo usa um operador que
    // nao foi registrado no resolver - nao adianta mexer na arena.
    Serial.println("ERRO: AllocateTensors() falhou.");
    Serial.println("      Se o log acima cita 'Didn't find op for builtin "
                   "opcode', registre esse operador no MicroMutableOpResolver");
    Serial.println("      (e aumente o parametro de template). Caso contrario, "
                   "aumente kTensorArenaSize.");
    Serial.printf("      Arena atual: %d bytes.\n", kTensorArenaSize);
    haltOnError("      Regenere os artefatos com 'python build.py' se o modelo mudou.");
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Falha cedo e de forma explicita se o firmware e o modelo divergirem.
  if (model_input->type != kTfLiteInt8 || model_output->type != kTfLiteInt8) {
    haltOnError("ERRO: esperado modelo quantizado em int8.");
  }
  if (model_input->bytes != (size_t)kWindowSize) {
    Serial.printf("ERRO: tensor de entrada tem %u bytes, esperado %d.\n",
                  (unsigned)model_input->bytes, kWindowSize);
    haltOnError("Regenere os artefatos com 'python build.py'.");
  }

  Serial.printf("[TFL] Arena utilizada: %u de %d bytes\n",
                (unsigned)interpreter->arena_used_bytes(), kTensorArenaSize);
  Serial.printf("[TFL] Quantizacao entrada: scale=%.8f zero_point=%d\n",
                model_input->params.scale, model_input->params.zero_point);
  Serial.printf("[TFL] Normalizacao do treino: media=%.6f desvio=%.6f\n",
                kNormMean, kNormStd);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Serial.println();
  Serial.println("=== TinyML ESP32 - Diagnostico de Rolamentos (v2) ===");

  setupSensor();
  setupModel();

  // Aviso de dominio: o modelo foi treinado com CWRU a 12 kHz, mas o MPU6050
  // nao passa de 1 kHz. As janelas cobrem faixas de frequencia diferentes.
  const float band_ratio = (float)kTrainingSampleRateHz / (float)g_sample_rate_hz;
  Serial.println();
  Serial.printf("[BANDA] Treino: %lu Hz (Nyquist %lu Hz, janela de %.1f ms)\n",
                (unsigned long)kTrainingSampleRateHz,
                (unsigned long)(kTrainingSampleRateHz / 2),
                1000.0f * kWindowSize / kTrainingSampleRateHz);
  Serial.printf("[BANDA] ESP32:  %lu Hz (Nyquist %lu Hz, janela de %.1f ms)\n",
                (unsigned long)g_sample_rate_hz,
                (unsigned long)(g_sample_rate_hz / 2),
                1000.0f * kWindowSize / g_sample_rate_hz);
  if (band_ratio > 1.05f || band_ratio < 0.95f) {
    Serial.printf("        [ALERTA] Divergencia de %.1fx. O modelo ve conteudo "
                  "espectral diferente do que foi treinado.\n",
                  band_ratio);
    Serial.println("        Correcao: ajustar TARGET_SAMPLE_RATE_HZ no build.py "
                   "e retreinar, ou usar o comando 'f' para casar a taxa.");
  } else {
    Serial.println("        [OK] Treino e aquisicao na mesma banda.");
  }

  printHelp();
  Serial.println("Inicializacao concluida.");
  Serial.println();
}

// ===========================================================================
// Loop principal
// ===========================================================================

void loop() {
  handleSerialCommand();

  if (g_paused) {
    delay(100);
    return;
  }

  g_window_count++;

  // ---- 1. Aquisicao -------------------------------------------------------
  SamplingReport rep;
  sampleWindow(rep);

  // ---- 2. Montagem do sinal de entrada -----------------------------------
  const float signal_std = buildInputSignal();

  // ---- 3. Normalizacao + quantizacao -------------------------------------
  const float scale = model_input->params.scale;
  const int zero_point = model_input->params.zero_point;

  int8_t q_min = 127;
  int8_t q_max = -128;
  uint32_t q_clipped = 0;
  uint8_t level_seen[32] = {0};  // bitmap dos 256 niveis int8 usados

  for (int i = 0; i < kWindowSize; i++) {
    const float normalized = (accel_buffer[i] - kNormMean) / (kNormStd + 1e-8f);
    int32_t quantized = (int32_t)lroundf(normalized / scale) + zero_point;
    if (quantized < -128 || quantized > 127) {
      q_clipped++;
      quantized = constrain(quantized, -128, 127);
    }
    const int8_t q = (int8_t)quantized;
    model_input->data.int8[i] = q;
    if (q < q_min) q_min = q;
    if (q > q_max) q_max = q;
    const uint8_t idx = (uint8_t)((int)q + 128);
    level_seen[idx >> 3] |= (uint8_t)(1u << (idx & 7));
  }

  int distinct_levels = 0;
  for (int b = 0; b < 32; b++) {
    uint8_t v = level_seen[b];
    while (v) {
      distinct_levels += (v & 1);
      v >>= 1;
    }
  }

  // ---- 4. Inferencia ------------------------------------------------------
  const uint32_t start_us = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERRO: Invoke() falhou.");
    delay(kLoopDelayMs);
    return;
  }
  const uint32_t elapsed_us = micros() - start_us;

  // ---- 5. Relatorio -------------------------------------------------------
  const AxisStats sx = computeAxisStats(0);
  const AxisStats sy = computeAxisStats(1);
  const AxisStats sz = computeAxisStats(2);

  Serial.printf("--- Janela #%lu ---\n", (unsigned long)g_window_count);

  Serial.printf("[AMOSTRA] fs_alvo=%lu Hz | fs_real=%.1f Hz | dur=%.1f ms | "
                "jitter_max=%lu us | atrasos=%lu\n",
                (unsigned long)g_sample_rate_hz, rep.actual_fs_hz,
                rep.duration_us / 1000.0f, (unsigned long)rep.max_jitter_us,
                (unsigned long)rep.late_samples);

  Serial.printf("[SENSOR]  repetidas=%lu/%d (%.1f%%) | saturadas=%lu | "
                "erros_i2c=%lu\n",
                (unsigned long)rep.repeated_samples, kWindowSize,
                100.0f * rep.repeated_samples / kWindowSize,
                (unsigned long)rep.saturated_samples,
                (unsigned long)rep.i2c_errors);

  Serial.printf("[EIXOS g] X med=%+.4f dp=%.5f pico=%+.4f/%+.4f\n", sx.mean,
                sx.std, sx.min, sx.max);
  Serial.printf("          Y med=%+.4f dp=%.5f pico=%+.4f/%+.4f\n", sy.mean,
                sy.std, sy.min, sy.max);
  Serial.printf("          Z med=%+.4f dp=%.5f pico=%+.4f/%+.4f\n", sz.mean,
                sz.std, sz.min, sz.max);

  Serial.printf("[ENTRADA] eixo=%s | dc_removido=%s | dp=%.6f g | "
                "dp_treino=%.6f g | razao=%.3fx\n",
                axisName(g_axis), g_remove_dc ? "sim" : "nao", signal_std,
                kNormStd, signal_std / kNormStd);

  Serial.printf("[QUANT]   int8 min=%d max=%d | niveis_distintos=%d/256 | "
                "saturados=%lu\n",
                (int)q_min, (int)q_max, distinct_levels,
                (unsigned long)q_clipped);

  // Alertas: cada um aponta uma causa concreta de saida congelada.
  if (distinct_levels <= 2) {
    Serial.println("[!!] Entrada praticamente CONSTANTE apos a quantizacao. A "
                   "saida do modelo sera identica a cada janela.");
  }
  if (signal_std < kStaticSignalThresholdG) {
    Serial.printf("[!!] Sinal quase estatico (dp=%.6f g). Excite o sensor "
                  "(bata na bancada) ou aumente o ganho mecanico.\n",
                  signal_std);
  }
  if (rep.repeated_samples > (uint32_t)kWindowSize / 10) {
    Serial.println("[!!] Muitas amostras repetidas: fs pedida acima da taxa de "
                   "atualizacao do sensor. Reduza com o comando 'f'.");
  }
  if (rep.late_samples > 0) {
    Serial.println("[!!] Prazos de amostragem perdidos: a leitura I2C nao cabe "
                   "no intervalo. A janela nao tem taxa uniforme.");
  }
  if (rep.saturated_samples > 0) {
    Serial.println("[!!] Amostras no fundo de escala: aumente kAccelRange para "
                   "MPU6050_RANGE_4_G ou 8_G.");
  }
  // O quantizador de entrada foi calibrado sobre o CWRU. Um sinal muito mais
  // forte nao "estoura" o int8 aos poucos: ele achata contra -128/+127 e chega
  // ao modelo como onda quadrada. Qualquer classe predita nesse regime e um
  // artefato do ceifamento, nao uma leitura da vibracao.
  if (q_clipped > (uint32_t)kWindowSize / 10) {
    Serial.printf("[!!] Quantizador ceifou %lu de %d amostras (%.0f%%). O sinal "
                  "chega ACHATADO ao modelo - a classe predita nao tem valor.\n",
                  (unsigned long)q_clipped, kWindowSize,
                  100.0f * q_clipped / kWindowSize);
  }
  const float amplitude_ratio = signal_std / kNormStd;
  if (amplitude_ratio < 0.1f || amplitude_ratio > 2.0f) {
    Serial.printf("[!!] Amplitude %.1fx a do treino: o modelo opera fora da "
                  "faixa que conhece.\n", amplitude_ratio);
  }

  if (g_verbose) {
    Serial.print("[CRU LSB] ");
    for (int k = 0; k < 8; k++) {
      Serial.printf("%d ", raw_window[k][(int)(g_axis == AXIS_MAGNITUDE ? AXIS_Z : g_axis)]);
    }
    Serial.println();
    Serial.print("[ENTRADA g] ");
    for (int k = 0; k < 8; k++) {
      Serial.printf("%+.5f ", accel_buffer[k]);
    }
    Serial.println();
    Serial.print("[INT8] ");
    for (int k = 0; k < 8; k++) {
      Serial.printf("%d ", (int)model_input->data.int8[k]);
    }
    Serial.println();
  }

  int best_class = 0;
  float best_score = -1.0f;
  Serial.println("[SAIDA]");
  for (int i = 0; i < kNumClasses; i++) {
    const float score =
        (model_output->data.int8[i] - model_output->params.zero_point) *
        model_output->params.scale;
    Serial.printf("  %-14s %.4f (int8=%d)\n", kClassLabels[i], score,
                  (int)model_output->data.int8[i]);
    if (score > best_score) {
      best_score = score;
      best_class = i;
    }
  }

  Serial.printf("=> %s (%.1f%%) em %lu us\n\n", kClassLabels[best_class],
                best_score * 100.0f, (unsigned long)elapsed_us);

  delay(kLoopDelayMs);
}
