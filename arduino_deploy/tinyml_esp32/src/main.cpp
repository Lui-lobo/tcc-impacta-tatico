/*
 * Deploy TinyML no ESP32 - TCC Manutencao Preditiva de Rolamentos
 *
 * Classifica janelas de 512 amostras de vibracao em: normal, falha na pista
 * interna ou falha na pista externa, usando a CNN 1D quantizada em int8
 * gerada pelo pipeline Python (build.py).
 */
#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_params.h"
#include "model_tflite.h"

// Area de trabalho do interpretador (tensores intermediarios + metadados).
// O valor real consumido e impresso no setup(); ajuste se sobrar/faltar memoria.
constexpr int kTensorArenaSize = 24 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input = nullptr;
static TfLiteTensor* model_output = nullptr;

// Janela de aceleracao em unidades de 'g', como no dataset CWRU.
static float accel_buffer[kWindowSize];

static void haltOnError(const char* message) {
  Serial.println(message);
  while (true) {
    delay(1000);
  }
}

/*
 * TODO: substituir pela leitura real do acelerometro (MPU-6050 / ADXL345).
 * A taxa de amostragem precisa ser a mesma do CWRU (12 kHz) para que as
 * frequencias de falha caiam nas mesmas posicoes vistas no treino.
 */
static void readAccelerometerWindow(float* buffer, int length) {
  for (int i = 0; i < length; i++) {
    buffer[i] = (float)random(-2000, 2000) / 1000.0f;
  }
}

/*
 * ============================================================================
 * EXEMPLO DE INTEGRAÇÃO FUTURA COM MPU6050 (Mantido comentado a pedido do usuário)
 * ============================================================================
 * Para usar este código no futuro:
 * 1. Adicione "adafruit/Adafruit MPU6050" no seu arquivo platformio.ini (em lib_deps)
 * 2. Descomente as bibliotecas e variaveis globais abaixo
 * 3. Descomente a inicialização e coloque dentro do setup()
 * 4. Substitua a função readAccelerometerWindow atual por esta versão
 *
 * #include <Wire.h>
 * #include <Adafruit_MPU6050.h>
 * #include <Adafruit_Sensor.h>
 * 
 * Adafruit_MPU6050 mpu;
 * 
 * // Chame este trecho dentro da sua funcao setup():
 * // void setup_mpu() {
 * //   Wire.setClock(400000); // Configura o I2C em alta velocidade (400kHz)
 * //   if (!mpu.begin()) {
 * //     Serial.println("Falha ao encontrar o chip MPU6050");
 * //     while (1) { delay(10); }
 * //   }
 * //   mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
 * //   mpu.setFilterBandwidth(MPU6050_BAND_260_HZ); // Reduz atrasos no sensor
 * // }
 * 
 * // Esta será a função definitiva quando for usar o hardware real:
 * // static void readAccelerometerWindow(float* buffer, int length) {
 * //   sensors_event_t a, g, temp;
 * //   for (int i = 0; i < length; i++) {
 * //     mpu.getEvent(&a, &g, &temp);
 * //     
 * //     // O dataset CWRU usa forca G. O MPU6050 retorna m/s^2.
 * //     // Precisamos dividir por 9.81 para normalizar a escala para 'g'.
 * //     buffer[i] = a.acceleration.z / 9.81f; 
 * //     
 * //     // IMPORTANTE: Tentar atingir 12kHz reais via I2C é um desafio no ESP32.
 * //     // Voce pode adicionar um delayMicroseconds se precisar estabilizar a taxa,
 * //     // mas provavelmente o barramento I2C já sera o limitador de velocidade.
 * //   }
 * // }
 * ============================================================================
 */

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Serial.println();
  Serial.println("=== TinyML ESP32 - Diagnostico de Rolamentos ===");
  Serial.printf("Modelo na flash: %u bytes\n", model_tflite_len);

  const tflite::Model* model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    haltOnError("ERRO: schema do modelo incompativel com a versao da biblioteca.");
  }

  // Resolver enxuto: apenas os 6 operadores presentes neste modelo, o que
  // economiza flash em relacao ao AllOpsResolver.
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
    haltOnError("ERRO: AllocateTensors() falhou. Aumente kTensorArenaSize.");
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

  Serial.printf("Arena utilizada: %u de %d bytes\n",
                (unsigned)interpreter->arena_used_bytes(), kTensorArenaSize);
  Serial.printf("Quantizacao entrada: scale=%.8f zero_point=%d\n",
                model_input->params.scale, model_input->params.zero_point);
  Serial.println("Inicializacao concluida.");
  Serial.println();
}

void loop() {
  readAccelerometerWindow(accel_buffer, kWindowSize);

  // Mesma normalizacao estatistica do treino, seguida da quantizacao afim
  // exigida pelo tensor de entrada int8.
  const float scale = model_input->params.scale;
  const int zero_point = model_input->params.zero_point;

  for (int i = 0; i < kWindowSize; i++) {
    const float normalized = (accel_buffer[i] - kNormMean) / (kNormStd + 1e-8f);
    int32_t quantized = (int32_t)lroundf(normalized / scale) + zero_point;
    quantized = constrain(quantized, -128, 127);
    model_input->data.int8[i] = (int8_t)quantized;
  }

  const uint32_t start_us = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERRO: Invoke() falhou.");
    delay(2000);
    return;
  }
  const uint32_t elapsed_us = micros() - start_us;

  int best_class = 0;
  float best_score = -1.0f;
  for (int i = 0; i < kNumClasses; i++) {
    const float score =
        (model_output->data.int8[i] - model_output->params.zero_point) *
        model_output->params.scale;
    Serial.printf("  %-14s %.4f\n", kClassLabels[i], score);
    if (score > best_score) {
      best_score = score;
      best_class = i;
    }
  }

  Serial.printf("=> %s (%.1f%%) em %lu us\n\n", kClassLabels[best_class],
                best_score * 100.0f, (unsigned long)elapsed_us);

  delay(2000);
}
