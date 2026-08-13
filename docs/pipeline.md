# Como a Pipeline Funciona

Documento de referência do `build.py`: o que cada etapa faz, por que faz assim, o
que produz e como o resultado chega ao ESP32.

Uma execução completa vai do arquivo `.mat` bruto ao firmware compilável, em seis
etapas encadeadas. Nada é manual: o único comando é `python build.py`.

---

## Visão geral

```
data/*.mat  (12 kHz / 48 kHz, ~102 MB)
     │
     │  [1] carregar → decimar → cortar → janelar
     ▼
X_train (6.104 × 512)   X_test (1.313 × 512)
     │
     │  [2] treinar a CNN 1D  (Keras, float32)
     ▼
modelo.keras  ·  9.435 parâmetros
     │
     │  [3] quantizar  (post-training, int8)
     ▼
model_quantized.tflite  ·  15,2 KB
     │
     ├──[4]──► model_tflite.cpp + model_params.h
     │
     ├──[5]──► test_vectors.cpp  (validação embarcada)
     │
     └──[6]──► relatorios/relatorio_metricas.md + PNGs
                     │
                     ▼
              pio run -t upload  →  ESP32
```

Cada etapa imprime `[n/6]` no console, então a saída do terminal acompanha esta
documentação linha a linha.

---

## Etapa 0 — Dataset (`download_cwru.py`)

Não faz parte do `build.py`, mas é pré-requisito.

Baixa 28 arquivos do CWRU Bearing Data Center para `data/<classe>/`:

| Classe | Arquivos | Taxa original | Cobertura |
|---|---|---|---|
| `0_normal` | 97–100 | 48 kHz | 4 cargas (0–3 hp) |
| `1_inner_race` | 105–108, 169–172, 209–212 | 12 kHz | 4 cargas × 3 diâmetros |
| `2_outer_race` | 130–133, 197–200, 234–237 | 12 kHz | 4 cargas × 3 diâmetros |

O rótulo da classe vem do **nome da pasta**: `int(pasta.split('_')[0])`. Pastas
cujo prefixo não é numérico (como `audio_simulation`) são ignoradas. Para
acrescentar uma classe, basta criar `3_ball_fault/` e popular — o pipeline se
ajusta sozinho, inclusive o número de saídas da rede.

O download é idempotente e escreve em `.parcial` antes de renomear, para que uma
interrupção não deixe um `.mat` truncado que o `scipy` tentaria carregar depois.

---

## Etapa 1 — Carregar, decimar, cortar e janelar

`load_datasets()` no `build.py`, apoiado em `src/pipeline/data_processor.py`.

### 1.1 Leitura
`load_mat_file()` procura a chave que contém `_DE_time` (*Drive End*, o
acelerômetro montado no mancal do lado do acionamento). Se não achar, cai para a
maior matriz do arquivo e avisa.

### 1.2 Decimação para 1 kHz — a decisão mais importante do projeto

O CWRU foi gravado a 12 kHz (falhas) e 48 kHz (baseline). **O MPU6050 atualiza
os registradores do acelerômetro a no máximo 1 kHz**, e não existe configuração
que mude isso.

Havia duas saídas, e a escolhida é a segunda:

| Opção | Consequência |
|---|---|
| Treinar a 12 kHz e inferir a 1 kHz | O modelo veria um conteúdo espectral que nunca aprendeu. Um impacto real pareceria 12x mais rápido. |
| **Decimar o dataset para 1 kHz e treinar nessa taxa** | O modelo aprende exatamente a banda que o sensor consegue entregar. |

`resample_to()` faz isso em **etapas fatoradas** (12 → 6×2; 48 → 8×6) em vez de
um único fator grande, porque `scipy.signal.decimate` degrada com fatores altos:
o filtro anti-aliasing precisaria de uma transição estreita demais e introduziria
ondulação na banda passante.

Usa FIR com `zero_phase=True` — filtra para frente e para trás, cancelando o
atraso de fase. **Preservar a fase importa** porque as assinaturas de falha são
impulsos periódicos, e um filtro de fase não linear distorceria justamente o
formato desses impulsos.

> **Custo a declarar:** a decimação descarta tudo acima de 500 Hz, incluindo a
> banda de ressonância de 2–5 kHz classicamente usada em diagnóstico de
> rolamentos. É a causa provável da falha a 0 hp documentada no
> [protocolo de validação](protocolo_validacao.md) §6.

O mapa `SOURCE_RATE_BY_FILE` marca 97–100 como 48 kHz. **Sem essa distinção, a
classe normal chegaria ao modelo numa escala de frequência 4x diferente das
classes de falha — e a rede aprenderia a taxa de amostragem em vez da falha.**

### 1.3 Corte treino/teste ANTES do janelamento

`split_time_series()` corta a série temporal em dois trechos contíguos, 80/20.

Esta ordem não é detalhe. Com sobreposição de 93,75%, janelas vizinhas
compartilham 480 das 512 amostras. Um `train_test_split` aleatório **depois** do
janelamento colocaria janelas quase idênticas dos dois lados, e a acurácia medida
seria vazamento puro.

Há ainda uma **zona morta de uma janela inteira** na fronteira, descontada do lado
do treino:

```
|<--------- treino --------->|<-- morta -->|<--- teste --->|
0                      cut-512           cut              n
```

A zona morta sai do treino porque descontá-la do teste encolheria demais o
conjunto de avaliação nos arquivos curtos.

> **Limite conhecido:** o corte é feito *dentro de cada arquivo*, então cada
> janela de teste tem uma contraparte de treino da mesma gravação. Não há
> vazamento de amostras, mas há de gravação. Ver
> [protocolo de validação](protocolo_validacao.md) §6.

### 1.4 Janelamento

`create_windows()` fatia em janelas de 512 amostras com passo de 32
(`overlap = 0.9375`). A sobreposição alta recupera a contagem de janelas perdida
na decimação — **sem criar informação nova**, razão pela qual o console também
imprime a contagem de janelas *independentes*:

```
Sinal total apos decimacao: 279371 amostras (279.4 s) de 28 arquivos
=> 545 janelas INDEPENDENTES (sem sobreposicao).
```

**É o 545 que limita a validade estatística, não os 7.417.**

### 1.5 Normalização

Média e desvio saem **apenas do treino** e são aplicados aos dois conjuntos, para
que o teste permaneça inédito:

```python
mean = np.mean(X_train); std = np.std(X_train)
X_train = (X_train - mean) / (std + 1e-8)
X_test  = (X_test  - mean) / (std + 1e-8)
```

Esses dois números viajam para o firmware em `model_params.h`. Uma cópia de
`X_test` em unidades físicas é guardada antes da normalização — é ela que vai
para a flash na etapa 5.

---

## Etapa 2 — Treinar a CNN 1D

`src/pipeline/model_builder.py`. Arquitetura de 9.435 parâmetros:

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

Três decisões guiam o desenho:

**Convolução direta sobre o sinal temporal.** Não há FFT em lugar nenhum. Os
filtros convolutivos extraem as features do array bruto, o que elimina a
necessidade de rodar DSP pesado no ESP32 antes da inferência.

**Redução dimensional agressiva e precoce.** `strides=2` combinado com
`MaxPooling` derruba 512 → 256 → 128 → 64 → 32 nas quatro primeiras camadas. Isso
encolhe os tensores intermediários, que são exatamente o que a *tensor arena* do
TFLite Micro precisa alocar. O resultado medido no hardware: **4.716 bytes de
arena**.

**O gargalo de parâmetros é o `Dense`.** 8.208 dos 9.435 parâmetros (87%) estão
na camada densa, alimentada pelo `Flatten` de 512 valores. É o alvo óbvio caso
seja preciso encolher mais o modelo.

### Balanceamento e critério de parada

Os arquivos têm durações diferentes e a classe normal tem só 4 arquivos (o CWRU
não oferece mais), então há desequilíbrio inerente. `class_weight` compensa:

```python
peso[i] = len(y_train) / (num_classes * contagem[i])
```

`EarlyStopping(patience=12, restore_best_weights=True)` evita gravar no ESP32 o
modelo da última época — uma escolha arbitrária que pode cair num vale ruim.

> **Ressalva metodológica:** não há dados para um terceiro conjunto, então o
> `EarlyStopping` monitora o próprio conjunto de teste. A acurácia reportada é,
> portanto, otimista. O `tools/validacao_por_carga.py` não tem esse defeito.

---

## Etapa 3 — Quantizar para int8

`src/pipeline/quantizer.py`. *Post-training integer quantization*: pesos e
ativações passam de float32 (4 bytes) para int8 (1 byte).

Ganho duplo: o modelo fica ~75% menor em flash, e o Xtensa faz aritmética inteira
nativamente, sem gastar ciclos em FPU.

### O dataset representativo

O conversor precisa observar ativações reais para calcular escala e zero-point de
cada tensor. `_representative_data_gen()` sorteia 150 janelas do treino.

Daí vem uma propriedade importante: **a escala de entrada é calibrada sobre a
amplitude do CWRU**. Sinais muito mais fortes não "estouram" gradualmente — eles
achatam contra −128/+127 e chegam ao modelo como onda quadrada. O firmware
detecta e alerta.

### O truque do batch fixo

```python
def _static_batch_model(self):
    batch_shape = (1,) + tuple(self.model.input_shape[1:])
    inputs = tf.keras.Input(batch_shape=batch_shape)
    return tf.keras.Model(inputs, self.model(inputs))
```

Com batch dinâmico (`None`), o `Flatten` do Keras 3 vira uma sequência
`SHAPE → STRIDED_SLICE → PACK → RESHAPE`: o grafo calcula a forma de saída em
tempo de execução. No TFLite Micro isso **quebra o boot** com
`Didn't find op for builtin opcode 'SHAPE'`, porque o interpretador só trabalha
com memória estática.

Fixando o batch em 1 — que é exatamente como o ESP32 infere, uma janela por vez —
a forma passa a ser conhecida na conversão e o TFLite dobra as quatro operações
numa constante.

**Efeito medido:** 24 operadores → 9; modelo de 17,76 KB → 15,24 KB.

O wrapper reutiliza as mesmas camadas, então os pesos treinados são
compartilhados, não copiados.

---

## Etapa 4 — Gerar os artefatos C

`src/pipeline/c_generator.py`. Produz dois pares de arquivos.

### `model_tflite.h` / `model_tflite.cpp`

O `.tflite` vira um array C. O array fica no `.cpp` e não no `.h` para que o
modelo possa ser incluído por várias unidades de compilação sem gerar símbolos
duplicados na linkagem.

```cpp
alignas(16) const unsigned char model_tflite[] = { 0x1c, 0x00, ... };
```

O `alignas(16)` é **obrigatório**: o flatbuffer do TFLite Micro é lido
diretamente da flash, sem cópia, e exige esse alinhamento.

### `model_params.h` — o contrato entre Python e firmware

```c
constexpr int   kWindowSize           = 512;
constexpr int   kNumClasses           = 3;
constexpr int   kTrainingSampleRateHz = 1000;
constexpr float kNormMean             = 0.0080134990f;
constexpr float kNormStd              = 0.0345664136f;
static const char* const kClassLabels[] = {"0_normal", "1_inner_race", "2_outer_race"};
```

Este arquivo existe para que **firmware e treino nunca divirjam**. O
`kTrainingSampleRateHz` alimenta diretamente a taxa de amostragem do firmware:

```cpp
constexpr uint32_t kDefaultSampleRateHz = (uint32_t)kTrainingSampleRateHz;
```

Mudar `TARGET_SAMPLE_RATE_HZ` no `build.py` e regravar já reconfigura o divisor
interno do MPU6050 — não há constante duplicada para esquecer de atualizar.

---

## Etapa 5 — Exportar o conjunto de validação embarcada

Grava na flash as mesmas janelas de teste que produziram a acurácia do relatório,
junto com a saída de referência do interpretador do PC. O comando `t` no console
serial as reprocessa dentro do ESP32 e compara.

Quatro decisões sustentam o rigor do ensaio:

**Janelas exportadas antes da normalização**, em unidades físicas. O ESP32 aplica
a própria normalização e a própria quantização, então o teste exercita a cadeia
inteira e não apenas o `Invoke()`.

**Armazenamento em `int16`** com escala única. São ~15 bits úteis contra os 8 que
o quantizador preserva, então nada que o modelo enxergue se perde, e o custo em
flash cai pela metade. É também o formato nativo do MPU6050.

**Referência com os kernels `BUILTIN_REF`**, que são os que o TFLite Micro
implementa — não os otimizados de desktop (XNNPACK/ruy). Ambos foram verificados
e produzem saída idêntica, mas a escolha mantém a comparação correta.

**Hash FNV-1a da janela já quantizada.** Permite ao ESP32 provar que montou a
mesma entrada. Sem isso, uma divergência na saída seria ambígua entre erro de
pré-processamento e diferença de kernels. Com isso, o ensaio isola a causa.

Uma sutileza fácil de errar: `round_half_away()` replica o `lroundf()` da libc do
ESP32, que arredonda afastando do zero. O `np.round()` usa arredondamento
bancário e acusaria divergências de 1 LSB que não existem.

`MAX_TEST_VECTORS = 128` limita o custo em flash (≈129 KB); acima disso a seleção
é estratificada e preserva a proporção entre as classes.

---

## Etapa 6 — Relatório e gráficos

`src/pipeline/evaluator.py` gera `relatorios/relatorio_metricas.md` com duas
acurácias lado a lado — Keras float32 e **TFLite int8, que é o que roda no
ESP32** —, a escala do conjunto de dados, precisão/recall de ambos e os limites
de hardware. As matrizes de confusão float32 e int8 saem no mesmo PNG a 300 DPI.

O relatório traz também o aviso sobre como ler a acurácia, apontando para o
`tools/validacao_por_carga.py`.

---

## Da pipeline ao firmware

O `build.py` escreve direto na árvore do projeto PlatformIO:

```
arduino_deploy/tinyml_esp32/
├── include/
│   ├── model_params.h      ← etapa 4
│   ├── model_tflite.h      ← etapa 4
│   └── test_vectors.h      ← etapa 5
├── src/
│   ├── main.cpp            ← escrito à mão
│   ├── model_tflite.cpp    ← etapa 4  (gerado)
│   └── test_vectors.cpp    ← etapa 5  (gerado)
└── platformio.ini
```

Depois é só `pio run -t upload`. O PlatformIO compila todos os `.cpp` de `src/`,
resolve as bibliotecas de `lib_deps` e grava.

Dois pontos do `platformio.ini` merecem atenção:

```ini
build_unflags = -std=gnu++11
build_flags   = -std=gnu++17 -DTF_LITE_STATIC_MEMORY
```

O TensorFlow Lite Micro exige C++17, e o core Arduino-ESP32 2.x compila em
gnu++11 por padrão. Sem o `build_unflags`, a compilação falha.

### Como o firmware consome os artefatos

```cpp
static tflite::MicroMutableOpResolver<6> resolver;   // resolver enxuto
resolver.AddConv2D(); resolver.AddMaxPool2D(); ...   // 5 ops + 1 de reserva
```

Um `MicroMutableOpResolver` com os operadores exatos, em vez do `AllOpsResolver`,
economiza flash. Se algum dia o boot acusar `Didn't find op for builtin opcode`,
o conserto é registrar o operador ali e aumentar o parâmetro do template — não
mexer no tamanho da arena.

A arena é estática e dimensionada com folga:

```cpp
constexpr int kTensorArenaSize = 24 * 1024;   // uso real medido: 4.716 bytes
```

O caminho de inferência a cada janela, em `main.cpp`:

```
sampleWindow()          → 512 amostras do MPU6050, cronograma absoluto
buildInputSignal()      → escolhe o eixo, converte para g, REMOVE O DC
quantizeInputWindow()   → (x − kNormMean) / kNormStd → int8
interpreter->Invoke()   → 9,3 ms
desquantiza a saída     → (int8 − zero_point) × scale
```

A remoção de DC é indispensável e foi a causa do primeiro bug do projeto: o eixo
Z em repouso vale ~1 g constante, enquanto o CWRU é um sinal AC de média ≈ 0. Sem
removê-la, o vetor de entrada virava uma constante após a normalização e a saída
do modelo ficava congelada no mesmo valor a cada janela.

`quantizeInputWindow()` é **a mesma função** usada pelo laço principal e pela
suíte de validação. Uma implementação só — se fossem duas cópias, a validação
poderia aprovar um caminho que não é o que roda em campo.

---

## Reprodutibilidade

`tf.keras.utils.set_random_seed(SEED)` no início do `main()` fixa NumPy, `random`
e TensorFlow de uma vez. Com a mesma entrada, o pipeline gera artefatos **byte a
byte idênticos** — verificável com `git status` após reexecutar: se nada mudou
nos dados, `model_tflite.cpp` não aparece como modificado.

Isso importa porque quem avaliar o trabalho precisa conseguir chegar aos mesmos
números.

---

## Onde mexer

| Quero mudar | Constante | Arquivo |
|---|---|---|
| Taxa de amostragem | `TARGET_SAMPLE_RATE_HZ` | `build.py` |
| Tamanho da janela | `WINDOW_SIZE` | `build.py` |
| Sobreposição | `WINDOW_OVERLAP` | `build.py` |
| Proporção de teste | `TEST_RATIO` | `build.py` |
| Épocas / batch | `EPOCHS`, `BATCH_SIZE` | `build.py` |
| Vetores gravados na flash | `MAX_TEST_VECTORS` | `build.py` |
| Arquivos baixados | `CATALOGO` | `download_cwru.py` |
| Arquitetura da rede | `build_cnn1d()` | `src/pipeline/model_builder.py` |
| Eixo, fundo de escala, arena | bloco de configuração | `arduino_deploy/.../main.cpp` |

Depois de mexer em qualquer constante do `build.py`, rode `python build.py` e
`pio run -t upload`: os artefatos gerados carregam os valores novos
automaticamente.

---

## Documentos relacionados

- **[`protocolo_validacao.md`](protocolo_validacao.md)** — o que é validado, como
  reproduzir cada ensaio e os resultados medidos no hardware.
- **[`../tinyml_relatorio.md`](../tinyml_relatorio.md)** — justificativa das
  decisões de arquitetura e as especificações formais dos tensores.
- **[`../README.md`](../README.md)** — início rápido.
