# Relatório Técnico: Pipeline TinyML para Manutenção Preditiva (ESP32)

Este documento justifica as decisões arquiteturais da IA e as instruções de integração técnica do modelo preditivo para inferência em microcontroladores com recursos criticamente escassos.

---

## 1. Organização do Pipeline e Paradigma SOLID

O código Python de treinamento foi abstraído rigorosamente conforme os princípios da Responsabilidade Única (SRP):
1. **`DataProcessor`**: Encarregada isoladamente de extrair os `.mat` brutos, realizar a leitura estrutural da série temporal e a vetorização matricial (*windowing*).
2. **`TinyMLModelBuilder`**: Camada que esconde as complexidades do Keras, sendo uma "fábrica" de redes focadas em *footprints* minúsculos (focando nos constraints de RAM e Flash).
3. **`ModelQuantizer`**: Abstrai todo o escopo do TensorFlow Lite Converter, lidando puramente com calibração e quantização estática do modelo.
4. **`CArtifactGenerator`**: Ponte terminal que isola as dependências de IO em C, transformando o binário quantizado para linguagem de máquina compatível com o compilador Xtensa do ESP32.

---

## 2. Decisões de Arquitetura e Trade-offs

### Fatiamento de Janela de Tempo (*Windowing*)
Optou-se tecnicamente pela extração de janelas de **512 amostras** com **50% de sobreposição (overlap)**.
* **Trade-off Envolvido (Acurácia vs RAM):** Se operássemos com janelas de 2048 pontos (ideais para capturar harmônicas longas em sub-frequências usando Fast Fourier Transform), esgotaríamos o uso contínuo de RAM dinâmico no ESP32. Ao utilizar uma janela temporal de `512` em frequências de base do CWRU (12 kHz), capturamos blocos de `~42 ms`. A sobreposição garante que vibrações repentinas geradas pelo impacto de esferas rachadas não sejam ignoradas entre coletas. O tamanho do tensor resultante `[512, 1]` em Int8 consome míseros **512 Bytes** brutos, um consumo quase imperceptível de SRAM.

### Arquitetura Base da Rede CNN 1D
Embutir Deep Learning em controladores como o ESP32 exige sacrifícios de profundidade e número de parâmetros:
* Adotou-se **Convoluções Rápidas em Espaço Temporal**, eliminando a necessidade de passar o sinal por funções pesadas e morosas de DSP (como FFTs pré-calculadas no ESP32). Os filtros convolutivos inferem as features direto do array temporal bruto.
* O uso agressivo de `strides=2` e blocos rápidos de `MaxPooling` reduzem violentamente a dimensão do tensor propagado precocemente na rede, o que minimiza a alocação exigida na **Tensor Arena** (O banco de memória do TF Lite Micro que rege a sobrevida da alocação das matrizes internas de ativação).

### Integer Quantization
* O modelo sofre uma compressão que transforma Float32 (4 bytes por peso/tensor) para **INT8 (1 byte)** via *Post-Training Quantization*.
* **Vantagem em Embarcados:** Processadores low-cost demoram ciclos preciosos flutuando em FPU. A álgebra de inteiros de 8 bits é nativa, super veloz e resulta numa IA `~75%` mais leve em Flash Storage final do que o modelo de origem.

---

## 3. Especificações Formais dos Tensores TFLite

Para realizar a vinculação do modelo no seu C++ com **TensorFlow Lite for Microcontrollers**, a estrutura exigida será:

* **Tensor de Entrada (`model_input`)**
  * **Shape (Formato):** `[1, 512, 1]` (*1 batch, 512 de tempo, 1 canal*).
  * **Data Type:** `Int8` (Com sinal `-128 a 127`).
  * **Peso Ram:** 512 Bytes (+ metadados).

* **Tensor de Saída (`model_output`)**
  * **Shape (Formato):** `[1, N]` (Onde `N` é a contagem de Classes de falha do Dataset).
  * **Data Type:** `Int8`. (O Array precisará passar por uma desquantização interna via software C++ para você obter a probabilidade flutuante `0.0 a 1.0` de cada nó final).

---

## 4. Integração Prática: Consumo C++ do Hardware Acelerômetro

Na etapa de consumo C++ no **ESP32** com seu sensor físico (seja um sensor analógico puro ou um MPU digital via barramento), respeite rigorosamente estes aspectos sob pena de falha sistêmica da Inteligência Artificial:

### A) Equivalência de Sampling Rate (Critico)
O modelo aprendeu as assinaturas físicas operando os arquivos MAT sob a taxa de `12 kHz` ou `48 kHz` do CWRU.
* **Sua Ação:** Modifique a taxa de amostragem física de seu sensor (ou da task RTOS via interrupção de timer de ADC) para reproduzir a exata frequência temporal em que a rede foi treinada. Se seu sensor emitir amostras a `1 kHz` enquanto o treino foi `12 kHz`, o que no acelerômetro aparenta ser um impacto lento de 1s, o ESP32 acreditará ser um sinal absurdamente rápido e anômalo.

### B) Escalonamento Bufferizado
Não jogue valores um-a-um para inferência.
1. Crie globalmente um `int8_t circular_buffer[512]`.
2. O sistema físico apenas deve preencher esse buffer continuamente.
3. No gatilho final em que 512 pontos forem atingidos, promova a cópia direta usando ponteiros (memcpy) para o ponteiro alocado no TFLM: `model_input->data.int8`.
4. Invoque de maneira síncrona: `interpreter->Invoke()`.

### C) Normalização e Parâmetros Base (Escala Int8)
Os acelerômetros geram leituras Float em Gs (-2g a 2g). Como nossa rede embarcada aguarda um array inteiro Int8 bruto:
* Descubra a Média (`mean`) e Desvio Padrão (`std`) de todo o array original em Python;
* Na placa, no ato de aquisição em hardware: subtraia a média e divida pelo desvio para padronizar (Z-score).
* Por fim, aplique a conta canônica exigida pelo framework TF Lite:
  `valor_inteiro = (float_padronizado / model_input->params.scale) + model_input->params.zero_point`.
* Só em posse desta conversão escalar base o processador poderá enviar os impulsos corretos do mundo real para o modelo matemático gerado no seu Cabeçalho `model_tflite.h`.
