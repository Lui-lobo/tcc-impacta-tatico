# Como a Pipeline Funciona

Documento de referência do `build.py`: o que cada etapa faz, por que faz assim, o
que produz e como o resultado chega ao ESP32.

Uma execução completa vai do arquivo `.mat` bruto ao firmware compilável, em seis
etapas encadeadas. Nada é manual: o único comando é `python build.py`.

---

## Visão geral

```
data/*.mat  (28 ensaios, 12 kHz / 48 kHz, ~102 MB)
     │
     │  [0.5] curadoria — OPCIONAL, mas usada quando presente
     │        python -m dataset_engineering.executar
     ▼
data_curado/*.npy + manifesto.json  (24 ensaios, já a 1 kHz)
     │
     │  [1] carregar → decimar → cortar → janelar
     ▼
X_train (5.214 × 512)   X_test (1.121 × 512)
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

> **As duas fontes possíveis.** O `build.py` usa `data_curado/` quando ele
> existe e cai de volta nos `.mat` brutos de `data/` quando não existe. Não há
> flag: a detecção é automática, e a **primeira linha da saída** diz qual fonte
> está em uso. Os números mudam conforme a fonte:
>
> | | Ensaios | Sinal | Treino / teste | Janelas independentes |
> |---|---|---|---|---|
> | `data/` (bruto) | 28 | 279,4 s | 6.104 / 1.313 | 545 |
> | `data_curado/` | 24 | 238,7 s | 5.214 / 1.121 | 466 |
>
> Todos os números deste documento se referem ao **dataset curado**, salvo
> indicação em contrário.

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

> Com o dataset curado em uso, o rótulo vem do **manifesto**, não do nome da
> pasta. Uma classe nova precisa entrar no catálogo de
> `dataset_engineering/config.py` e a curadoria precisa ser reexecutada — criar
> a pasta em `data/` sozinho não basta, porque o `build.py` não olha para lá
> enquanto `data_curado/` existir.

O download é idempotente e escreve em `.parcial` antes de renomear, para que uma
interrupção não deixe um `.mat` truncado que o `scipy` tentaria carregar depois.

---

## Etapa 0.5 — Curadoria (`dataset_engineering/`) — opcional

Também não faz parte do `build.py`. Roda em quatro etapas e produz `data_curado/`:

```bash
python -m dataset_engineering.executar
```

| Tratativa | O que corrige |
|---|---|
| **T1** | Seleciona o canal `_DE_time` pelo **número do arquivo**. `99.mat` carrega também as variáveis de `98.mat`, e a leitura ingênua entregava a série de 1 hp rotulada como 2 hp |
| **T2** | Aborta se faltar arquivo do catálogo, em vez de treinar em silêncio com dataset incompleto |
| **T3** | Deduplicação por hash da série |
| **T9** | Põe em quarentena os ensaios de falha cuja frequência característica não se destaca do ruído **nem no sinal original** — 4 dos 24 |

A saída é `data_curado/<classe>/<numero>.npy`, com a série **já decimada** para
1 kHz, mais um `manifesto.json` com classe, severidade, carga e rotação de cada
ensaio.

Guardar a série já decimada é deliberado: a decimação é determinística e cara, e
assim o `build.py`, o `tools/validacao_por_carga.py` e o próprio fluxo de
curadoria partem do **mesmo sinal, byte a byte**. Antes, cada ferramenta
decimava por conta própria.

Detalhes em [`../dataset_engineering/README.md`](../dataset_engineering/README.md);
o efeito medido de cada tratativa, em
[`comparativo_curadoria.md`](comparativo_curadoria.md).

---

## Etapa 1 — Carregar, decimar, cortar e janelar

`load_datasets()` no `build.py`, apoiado em `src/pipeline/data_processor.py`.

### 1.0 Descoberta da fonte

`descobrir_fontes()` resolve de onde vêm os dados e devolve uma lista uniforme
de ensaios, cada um com a série **já na taxa em que será usada**. O resto do
`load_datasets()` é indiferente a qual fonte está em uso.

| Fonte | Condição | Leitura | Decimação |
|---|---|---|---|
| `data_curado/` | `manifesto.json` existe | `src/pipeline/dataset_curado.py` | já feita na etapa 0.5 |
| `data/*.mat` | caso contrário | `DataProcessor.load_mat_file()` | feita aqui |

Na primeira, `resample_to()` vira uma operação nula — a série já está a 1 kHz e
`source_rate == target_rate`. O caminho de código é o mesmo; só não há trabalho a
fazer.

### 1.1 Leitura

**Fonte bruta.** `load_mat_file()` procura a **primeira** chave que contém
`_DE_time` (*Drive End*, o acelerômetro montado no mancal do lado do
acionamento). Se não achar nenhuma, cai para a maior matriz do arquivo e avisa.

> ⚠️ **Essa heurística tem um defeito conhecido.** Alguns arquivos do CWRU
> carregam as variáveis de mais de um ensaio: `99.mat` traz também as de
> `98.mat`, e a primeira chave `_DE_time` que aparece é `X098_DE_time`. Pela
> fonte bruta, portanto, **`99.mat` entrega a série de `98.mat`** — outro nível
> de carga, sem nenhum aviso.
>
> A tratativa **T1** da etapa 0.5 corrige isso derivando a chave do número do
> arquivo (`X099_DE_time`). O `load_mat_file()` foi mantido como está de
> propósito: ele é o caminho histórico, e alterá-lo mudaria silenciosamente os
> resultados de quem rodar sem curadoria. Quem quiser o dado correto usa
> `data_curado/`.

**Fonte curada.** A chave já foi resolvida na curadoria e está registrada no
manifesto, campo `chave_mat`. Não há heurística em tempo de treino.

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

> Na fonte curada esse mapa não é consultado: a taxa de origem de cada ensaio
> está no manifesto (`fs_origem_hz`) e a decimação já aconteceu. O
> `SOURCE_RATE_BY_FILE` continua no `build.py` por causa do caminho de fallback.

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
Fonte: dataset curado v2: 24 ensaios (4 em quarentena), 238.7 s a 1000 Hz
...
Sinal total apos decimacao: 238743 amostras (238.7 s) de 24 arquivos
=> 466 janelas INDEPENDENTES (sem sobreposicao).
```

**É o 466 que limita a validade estatística, não os 6.335.** (Sem curadoria:
545 contra 7.417.)

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

O desequilíbrio vem da **contagem de ensaios**, não da duração: quase todos têm
≈ 10 s (a exceção é `97.mat`, com 5,1 s), mas o CWRU só publica 4 ensaios de
rolamento saudável contra 12 por tipo de falha. Com a curadoria, a proporção
fica 4 / 12 / 8 ensaios — 14,8% / 51,1% / 34,1% do sinal. `class_weight`
compensa:

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

> `kNormMean` e `kNormStd` **mudam a cada execução com dataset diferente** — são
> a média e o desvio do conjunto de treino. Os valores acima são os do build sem
> curadoria; confira o arquivo real em vez de copiar daqui. O firmware imprime
> `dp_treino` no console a cada janela, o que permite identificar qual build está
> gravado.

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
implementa — não os otimizados de desktop (XNNPACK/ruy), que reordenam
acumulações e usam caminhos vetorizados. Comparar o ESP32 contra os otimizados
mediria a diferença entre duas implementações, não a fidelidade do deploy.

O `build.py` roda os **dois** conjuntos de kernels sobre as mesmas janelas e
imprime o quanto eles divergem entre si:

```
Divergencia entre os dois conjuntos de kernels no PC: N% identicas | desvio max = M LSB
```

Esse número é o **piso** da divergência que se veria no ESP32 caso a referência
gravada fosse a errada. Confira o valor da execução atual em vez de assumir que
os dois coincidem.

**Hash FNV-1a da janela já quantizada.** Permite ao ESP32 provar que montou a
mesma entrada. Sem isso, uma divergência na saída seria ambígua entre erro de
pré-processamento e diferença de kernels. Com isso, o ensaio isola a causa.

Uma sutileza fácil de errar: `round_half_away()` replica o `lroundf()` da libc do
ESP32, que arredonda afastando do zero. O `np.round()` usa arredondamento
bancário e acusaria divergências de 1 LSB que não existem.

`MAX_TEST_VECTORS = 128` limita o custo em flash (≈129 KB); acima disso a seleção
é estratificada e preserva a proporção entre as classes.

> **A distribuição das 128 janelas identifica o build.** Como a seleção é
> estratificada, ela é uma função direta do conjunto de teste:
>
> | Fonte | Teste por classe | 128 vetores gravados |
> |---|---|---|
> | `data/` (bruto) | 160 / 577 / 576 | **16 / 56 / 56** |
> | `data_curado/` | 160 / 577 / 384 | **18 / 66 / 44** |
>
> A matriz de confusão do comando `t` soma exatamente esses valores nas linhas.
> É a forma mais rápida de conferir se o firmware gravado corresponde ao build
> em questão — ver [`comparativo_curadoria.md`](comparativo_curadoria.md) §1.

---

## Etapa 6 — Relatório e gráficos

`src/pipeline/evaluator.py` gera `relatorios/relatorio_metricas.md` com duas
acurácias lado a lado — Keras float32 e **TFLite int8, que é o que roda no
ESP32** —, a escala do conjunto de dados, precisão/recall de ambos e os limites
de hardware. As matrizes de confusão float32 e int8 saem no mesmo PNG a 300 DPI.

A seção *Escala do Conjunto de Dados* abre com a linha **Fonte dos dados** e
inclui a **distribuição do teste por classe**. As duas existem para que o
relatório se autoidentifique: sem elas, não há como saber de qual dataset um
relatório antigo veio, nem conferir se uma captura do comando `t` corresponde a
ele.

O relatório traz também o aviso sobre como ler a acurácia, apontando para o
`tools/validacao_por_carga.py`.

> **`relatorios/` é sobrescrito a cada execução.** Antes de retreinar sobre um
> dataset diferente, arquive o relatório e os PNGs — foi o que se fez em
> [`../relatorios/sem_curadoria/`](../relatorios/sem_curadoria/) para preservar a
> linha de base sem curadoria.

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
resolver.AddExpandDims();   resolver.AddConv2D();
resolver.AddMaxPool2D();    resolver.AddReshape();
resolver.AddFullyConnected(); resolver.AddSoftmax();  // as 6 vagas em uso
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
interpreter->Invoke()   → 9,3 ms (validação) · 9,9 ms (laço principal)
desquantiza a saída     → (int8 − zero_point) × scale
```

Os dois cronômetros medem **exatamente a mesma coisa** — só o `Invoke()`, sem
amostragem nem pré-processamento (`main.cpp:1042` e `main.cpp:627`). A diferença
de ~600 µs entre eles é, portanto, do próprio kernel, não de trabalho extra.

A explicação provável é o **cache de instruções**: na validação o `Invoke()` roda
128 vezes seguidas e a cache fica quente, enquanto no laço principal ele roda uma
vez a cada 2 s, com amostragem I2C e I/O serial no intervalo. Não foi medido
diretamente — é hipótese, e uma forma barata de testar seria chamar `Invoke()`
duas vezes seguidas no laço e comparar os dois tempos.

Em qualquer dos casos o ciclo útil fica abaixo de 2% da janela de 512 ms.

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
| Catálogo dos ensaios (classe, carga, severidade) | `_TABELA` | `dataset_engineering/config.py` |
| Critério de quarentena | `PISO_ENVELOPE` | `dataset_engineering/config.py` |
| Arquitetura da rede | `build_cnn1d()` | `src/pipeline/model_builder.py` |
| Eixo, fundo de escala, arena | bloco de configuração | `arduino_deploy/.../main.cpp` |

Depois de mexer em qualquer constante do `build.py`, rode `python build.py` e
`pio run -t upload`: os artefatos gerados carregam os valores novos
automaticamente.

Depois de mexer em qualquer coisa do `dataset_engineering/config.py`, rode
`python -m dataset_engineering.executar` **antes** do `build.py` — o dataset
curado não se regenera sozinho.

Para voltar ao comportamento sem curadoria, apague `data_curado/`. Para treinar
com o dataset completo mas ainda com a leitura corrigida:

```bash
python -m dataset_engineering.executar --manter-suspeitos
```

---

## Documentos relacionados

- **[`dataset.md`](dataset.md)** — auditoria do dataset: o que a decimação para
  1 kHz descarta e por que ela é a limitação dominante do projeto.
- **[`../dataset_engineering/README.md`](../dataset_engineering/README.md)** — o
  fluxo da etapa 0.5, tratativa por tratativa.
- **[`comparativo_curadoria.md`](comparativo_curadoria.md)** — o sistema com e
  sem curadoria, nos três níveis de medição.
- **[`protocolo_validacao.md`](protocolo_validacao.md)** — o que é validado, como
  reproduzir cada ensaio e os resultados medidos no hardware.
- **[`ensaios_bancada.md`](ensaios_bancada.md)** — o comportamento do sistema
  fora da distribuição de treino, medido com excitação real.
- **[`../tinyml_relatorio.md`](../tinyml_relatorio.md)** — justificativa das
  decisões de arquitetura e as especificações formais dos tensores.
- **[`../README.md`](../README.md)** — início rápido.
