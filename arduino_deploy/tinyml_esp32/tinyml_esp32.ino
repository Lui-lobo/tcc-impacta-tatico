/*
 * Exemplo de Deploy para ESP32 - TCC Manutenção Preditiva
 * Pipeline TensorFlow Lite para Microcontroladores
 */
#include "model_tflite.h"

// Tamanho do tensor de entrada. Deve bater com o treinado em Python.
#define WINDOW_SIZE 512
int8_t accel_buffer[WINDOW_SIZE];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Inicializando TinyML no ESP32...");
  
  // Imprime os primeiros bytes do modelo quantizado
  Serial.print("Tamanho do modelo na Flash (Bytes): ");
  Serial.println(model_tflite_len);
  
  Serial.print("Assinatura de Memória (Hex): ");
  for(int i=0; i<4; i++){
    Serial.print(model_tflite[i], HEX);
    Serial.print(" ");
  }
  Serial.println();
}

void loop() {
  // Simulação de preenchimento do buffer através da leitura I2C/SPI de um MPU/ADXL
  for(int i = 0; i < WINDOW_SIZE; i++){
    // O int8_t captura oscilações de -128 a 127
    accel_buffer[i] = (int8_t)random(-128, 127); 
  }
  
  // No framework TFLite Micro, você faria a cópia do buffer físico para o tensor:
  // memcpy(model_input->data.int8, accel_buffer, WINDOW_SIZE);
  // interpreter->Invoke();
  
  Serial.println("Coleta simulada executada e Buffer preenchido.");
  delay(2000);
}
