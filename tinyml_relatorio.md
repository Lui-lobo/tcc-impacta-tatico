# Relatório Técnico: Pipeline TinyML para Manutenção Preditiva (ESP32)

Este documento justifica as decisões arquiteturais da IA e especifica a
integração do modelo preditivo em microcontroladores com recursos criticamente
escassos.

Para o passo a passo operacional da pipeline, ver
[`docs/pipeline.md`](docs/pipeline.md). Para os ensaios de validação e seus
resultados, ver [`docs/protocolo_validacao.md`](docs/protocolo_validacao.md).

---

## 1. Organização do Pipeline e Paradigma SOLID

O código Python de treinamento foi abstraído conforme o princípio da
Responsabilidade Única (SRP):

1. **`DataProcessor`** — extrai os `.mat` brutos, reamostra para a taxa de
   destino, corta treino/teste sem vazamento e fatia em janelas.
2. **`TinyMLModelBuilder`** — fábrica de redes com *footprint* mínimo, escondendo
   as complexidades do Keras sob os constraints de RAM e Flash.
3. **`ModelQuantizer`** — abstrai o TensorFlow Lite Converter: calibração,
   quantização estática e o wrapper de batch fixo exigido pelo TFLite Micro.
4. **`CArtifactGenerator`** — ponte terminal que isola as dependências de IO em
   C, transformando o binário quantizado em código compilável pelo Xtensa.
5. **`ModelEvaluator`** — métricas, gráficos e o relatório de consumo de
   hardware, incluindo a comparação float32 × int8.

---

## 2. Decisões de Arquitetura e Trade-offs

### 2.1 Taxa de amostragem: por que 1 kHz e não 12 kHz

Esta é a decisão que condiciona todas as outras.

O CWRU foi gravado a 12 kHz (falhas) e 48 kHz (baseline). **O MPU6050 atualiza os
registradores do acelerômetro a no máximo 1 kHz**, limite de hardware sem
configuração que o contorne.

Treinar na taxa original e inferir a 1 kHz colocaria o modelo diante de um
conteúdo espectral que ele nunca viu: um impacto real pareceria 12x mais rápido
do que a rede aprendeu. Optou-se, portanto, por **decimar o dataset para 1 kHz e
treinar nessa taxa**, de modo que o modelo aprenda exatamente a banda que o
sensor consegue entregar.

* **Trade-off assumido:** a decimação descarta tudo acima de 500 Hz, incluindo a
  banda de ressonância de 2–5 kHz classicamente usada em diagnóstico de
  rolamentos. É o custo de usar um MPU6050 em vez de um acelerômetro
  piezoelétrico com condicionamento dedicado, e é a causa provável da degradação
  observada com o rolamento descarregado.
* **Implementação:** decimação em etapas fatoradas (12 → 6×2; 48 → 8×6), com FIR
  de fase zero. Preservar a fase importa porque as assinaturas de falha são
  impulsos periódicos, e um filtro de fase não linear distorceria o formato
  desses impulsos.

### 2.2 Fatiamento de Janela de Tempo (*Windowing*)

Janelas de **512 amostras** com **93,75% de sobreposição** (passo de 32).

* **Trade-off (resolução espectral × RAM):** janelas de 2048 pontos dariam melhor
  resolução em baixa frequência, mas quadruplicariam os tensores intermediários
  na *tensor arena*. A 1 kHz, 512 amostras cobrem **512 ms** — tempo suficiente
  para conter dezenas de períodos das frequências de falha (BPFI ≈ 162 Hz,
  BPFO ≈ 107 Hz a 1797 rpm). O tensor `[512, 1]` em int8 ocupa **512 bytes**.
* **Por que sobreposição tão alta:** a decimação divide a contagem de amostras
  por 12 (ou 48). O passo curto recupera a contagem de janelas de treino. Ela
  **não cria informação nova** — o pipeline reporta separadamente as janelas
  independentes (545 no dataset atual), que são o que limita a validade
  estatística.
* **Só é seguro porque o corte treino/teste acontece ANTES do janelamento**, com
  uma zona morta de uma janela inteira na fronteira. Um `train_test_split`
  aleatório sobre janelas sobrepostas colocaria amostras quase idênticas dos dois
  lados.

### 2.3 Arquitetura Base da Rede CNN 1D

9.435 parâmetros no total:

| Camada | Saída | Parâmetros |
|---|---|---|
| `Conv1D(8, k=16, s=2, relu)` | 256 × 8 | 136 |
| `MaxPooling1D(2)` | 128 × 8 | 0 |
| `Conv1D(16, k=8, s=2, relu)` | 64 × 16 | 1.040 |
| `MaxPooling1D(2)` | 32 × 16 | 0 |
| `Flatten` | 512 | 0 |
| `Dense(16, relu)` | 16 | 8.208 |
| `Dropout(0.3)` | 16 | 0 |
| `Dense(3, softmax)` | 3 | 51 |

* **Convolução direta no domínio do tempo.** Não há FFT em lugar nenhum: os
  filtros convolutivos extraem as features do array temporal bruto, eliminando a
  necessidade de rodar DSP pesado no ESP32 antes de cada inferência.
* **Redução dimensional agressiva e precoce.** `strides=2` combinado com
  `MaxPooling` derruba 512 → 256 → 128 → 64 → 32 nas quatro primeiras camadas,
  encolhendo os tensores intermediários — que são exatamente o que a *tensor
  arena* do TFLite Micro precisa alocar. Resultado medido no hardware:
  **4.716 bytes de arena**, contra os 24 KB reservados.
* **O gargalo de parâmetros é a camada densa:** 8.208 dos 9.435 (87%). É o alvo
  óbvio caso seja necessário encolher mais o modelo.

### 2.4 Integer Quantization

*Post-training integer quantization*: pesos e ativações passam de float32
(4 bytes) para **int8** (1 byte).

* **Vantagem em embarcados:** o Xtensa faz aritmética inteira de 8 bits
  nativamente, sem gastar ciclos em FPU. O modelo final ocupa **15,24 KB** de
  flash, ~75% menos que o equivalente em float32.
* **Custo medido:** a quantização altera a classe de pouquíssimas janelas. O
  relatório gerado pelo `build.py` reporta as duas acurácias — float32 e int8 —
  lado a lado, justamente para tornar esse custo visível em vez de presumido.
* **Batch fixo em 1 na conversão.** Com batch dinâmico, o `Flatten` do Keras 3
  vira `SHAPE → STRIDED_SLICE → PACK → RESHAPE`, e o TFLite Micro — que só opera
  com memória estática — falha no boot com `Didn't find op for builtin opcode
  'SHAPE'`. Fixar o batch reduziu o grafo de 24 para 9 operadores e o modelo de
  17,76 KB para 15,24 KB.
* **A escala de entrada é calibrada sobre a amplitude do CWRU.** Sinais muito
  mais fortes não "estouram" gradualmente: achatam contra −128/+127 e chegam ao
  modelo como onda quadrada. O firmware detecta e alerta.

---

## 3. Especificações Formais dos Tensores TFLite

Estrutura exigida para vincular o modelo no C++ com **TensorFlow Lite for
Microcontrollers**:

* **Tensor de Entrada (`model_input`)**
  * **Shape:** `[1, 512, 1]` (1 batch, 512 no tempo, 1 canal)
  * **Data Type:** `int8` (com sinal, −128 a 127)
  * **Peso em RAM:** 512 bytes + metadados

* **Tensor de Saída (`model_output`)**
  * **Shape:** `[1, N]`, onde `N` é a contagem de classes
  * **Data Type:** `int8` — o array precisa passar por desquantização em software
    para virar probabilidade em ponto flutuante:
    `prob = (int8 − zero_point) × scale`

* **Operadores utilizados:** `RESHAPE`, `CONV_2D`, `MAX_POOL_2D`,
  `FULLY_CONNECTED`, `SOFTMAX`. Registram-se explicitamente num
  `MicroMutableOpResolver<6>` em vez do `AllOpsResolver`, para economizar flash.
  O `CONV_2D` aparece porque o conversor rebaixa `Conv1D` para `Conv2D`.

---

## 4. Integração Prática: Consumo C++ do Hardware Acelerômetro

Os quatro pontos abaixo são obrigatórios. Cada um deles, se ignorado, produz
inferência sistematicamente errada — e três deles causaram bugs reais neste
projeto.

### A) Equivalência de taxa de amostragem

O modelo aprende na taxa em que foi treinado. Neste projeto **o alinhamento foi
feito do lado do dataset**: o CWRU é decimado para 1 kHz, que é o teto do
MPU6050, em vez de tentar levar o sensor aos 12 kHz do dataset — o que é
fisicamente impossível.

* **Sua ação:** amostrar exatamente em `kTrainingSampleRateHz`, constante
  exportada em `model_params.h`. O firmware já a consome diretamente:

  ```cpp
  constexpr uint32_t kDefaultSampleRateHz = (uint32_t)kTrainingSampleRateHz;
  ```

  Assim não existe constante duplicada para esquecer de atualizar quando a taxa
  de treino mudar.
* **Amostrar com cronograma absoluto.** Cada amostra `i` deve sair em
  `t0 + i × intervalo`. Somar o intervalo a cada iteração acumula o custo da
  transação I2C e a janela termina com taxa efetiva menor que a pedida.
* **Verificar que o sensor acompanha.** O bit `DATA_RDY` (`INT_STATUS`, 0x3A)
  indica amostra nova. Ler mais rápido que a taxa interna do sensor apenas repete
  o valor anterior e destrói a forma de onda. O firmware conta essas repetições e
  alerta.

### B) Aquisição bufferizada

Não envie valores um a um para a inferência.

1. Preencha uma janela de `kWindowSize` amostras lendo apenas os **6 bytes do
   acelerômetro** (registrador `ACCEL_XOUT_H`, 0x3B), com `Wire` cru. O
   `getEvent()` da Adafruit transfere 14 bytes (accel + temp + giro) e converte
   para float dentro do laço crítico — ~55% mais tempo de barramento.
2. Só depois da janela completa, aplique o pré-processamento (item C) e escreva
   em `model_input->data.int8`.
3. Invoque de forma síncrona: `interpreter->Invoke()`.

Um `memcpy` direto dos valores brutos do sensor para o tensor **não funciona**:
faltam a remoção de DC, a normalização e a quantização.

### C) Remoção do nível DC — obrigatória

O eixo Z em repouso mede ~1,0 g constante por causa da gravidade. O dataset CWRU
é um sinal **AC de média ≈ 0**.

Sem remover o DC de cada janela, o vetor de entrada vira praticamente uma
constante após a normalização, e a saída do modelo **congela no mesmo valor a
cada iteração**. Foi exatamente o sintoma do primeiro bug do projeto: inferências
sucessivas devolvendo `0.9961` idêntico, variando apenas o tempo de execução.

```cpp
const float dc = media(janela);
for (int i = 0; i < kWindowSize; i++) janela[i] -= dc;
```

### D) Normalização e conversão para int8

Média e desvio saem **apenas do conjunto de treino** e são exportados em
`model_params.h` — não os recalcule na placa.

```cpp
const float normalizado = (amostra_g - kNormMean) / (kNormStd + 1e-8f);
int32_t q = (int32_t)lroundf(normalizado / model_input->params.scale)
          + model_input->params.zero_point;
model_input->data.int8[i] = (int8_t)constrain(q, -128, 127);
```

Dois detalhes que custam caro se ignorados:

* **`lroundf` arredonda afastando do zero.** O `np.round()` do Python usa
  arredondamento bancário. Ao comparar a saída do microcontrolador com a do PC,
  essa diferença produz divergências de 1 LSB que não existem no modelo. O
  pipeline usa `round_half_away()` para reproduzir a semântica da libc.
* **Monitore o ceifamento.** Contar quantas amostras bateram em −128/+127 é a
  única forma de saber que o sinal chegou achatado ao modelo. Nessa condição a
  classe predita é artefato do ceifamento, não leitura da vibração.

### E) Desquantização da saída

```cpp
float prob = (model_output->data.int8[i] - model_output->params.zero_point)
           * model_output->params.scale;
```

Com `scale = 1/256` e `zero_point = −128`, a confiança máxima representável é
`0,9961` (`int8 = 127`). **Um valor de exatamente 0,9961 significa saturação**, e
`0,5000` (`int8 = 0`) significa empate perfeito entre duas classes.

---

## 5. Consumo de Recursos Medido

Valores obtidos no hardware, não estimados:

| Recurso | Valor | Total disponível |
|---|---|---|
| Modelo em flash | 15,24 KB (15.608 B) | — |
| Firmware completo (com vetores de teste) | 508,6 KB (520.801 B) | 1,25 MB — 39,7% |
| Tensor arena | 4,6 KB (4.716 B) | 24 KB reservados — 19,2% |
| RAM total do firmware | 51,0 KB (52.256 B) | 320 KB — 15,9% |
| Tempo de inferência | 9,3 ms | — |
| Ciclo útil (janela de 512 ms) | 1,82% | — |

O gargalo do sistema é a **aquisição**, não a inferência: encher a janela leva
512 ms e a classificação consome 1,8% desse tempo. Sobra folga para telemetria,
Wi-Fi ou OTA.
