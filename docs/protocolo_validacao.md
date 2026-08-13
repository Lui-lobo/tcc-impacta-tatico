# Protocolo de Validação do Sistema Embarcado

Documento de procedimento. Descreve **o que é validado, como reproduzir e como
interpretar** cada ensaio do sistema ESP32 + MPU6050 para diagnóstico de
rolamentos. Os números registrados aqui foram obtidos no hardware real e devem
ser refeitos após qualquer alteração no `build.py` ou no firmware.

---

## 1. O problema de validação

O modelo foi treinado com o dataset CWRU: sinais de acelerômetro de um rolamento
girando a 1797 rpm, com defeitos usinados na pista interna e na pista externa. As
classes são definidas por frequências características do defeito — BPFI ≈ 162 Hz
e BPFO ≈ 107 Hz — que **só existem quando há um eixo girando sob carga**.

Sobre uma bancada de protoboard não há rolamento. Portanto:

> Nenhum ensaio de mesa pode validar se o modelo *classifica corretamente uma
> falha de rolamento*. Qualquer classe predita nessa condição é ruído.

O que **pode** ser validado sem bancada rotativa são as duas metades restantes do
sistema: a aquisição do sinal e a fidelidade da inferência embarcada. O protocolo
separa os ensaios exatamente nessas camadas.

| Nível | O que valida | Ensaio | Situação |
|---|---|---|---|
| **V1** | Aquisição: I2C, taxa de amostragem, jitter, resolução | Excitação mecânica externa | ✅ aprovado |
| **V2** | Inferência embarcada: normalização, quantização, aritmética int8 | Injeção do conjunto de teste (`t`) | ✅ aprovado |
| **V3** | Generalização para rolamento físico | Bancada com eixo girando | ❌ fora do escopo |

---

## 2. Ensaio V1 — Cadeia de aquisição

### Objetivo
Comprovar que o ESP32 amostra o MPU6050 a uma taxa determinística e que a forma
de onda adquirida corresponde a uma excitação física conhecida.

### Procedimento
1. Gravar o firmware: `cd arduino_deploy/tinyml_esp32 && pio run -t upload`.
2. Abrir o monitor serial a 115200 bps. O diagnóstico de boot roda sozinho; pode
   ser repetido com o comando `s`.
3. Aplicar uma fonte de vibração periódica ao suporte do sensor. **Motor de
   vibração de celular** funciona bem: é periódico, repetível e tem frequência
   dentro da banda útil (Nyquist = 500 Hz).
4. Com a vibração ativa, capturar uma janela em CSV com o comando `r`.
5. Calcular a FFT do CSV e localizar o pico.

### Critérios de aceitação

| Métrica | Critério | Medido |
|---|---|---|
| `WHO_AM_I` | 0x68 | 0x68 ✅ |
| Custo de leitura I2C (6 bytes) | < 1000 µs | 281 µs ✅ |
| ODR real do sensor (via `DATA_RDY`) | = fs alvo | 1000 Hz ✅ |
| `fs_real` da janela | dentro de ±1% do alvo | 999,5 Hz ✅ |
| `jitter_max` | < 5% do intervalo (50 µs) | 13 µs ✅ |
| `repetidas` | 0 | 0/512 ✅ |
| `erros_i2c` / `atrasos` | 0 | 0 / 0 ✅ |

### Interpretação
A cadeia de aquisição está validada. O sistema amostra a 1 kHz com desvio máximo
de 13 µs — 1,3% do intervalo entre amostras — e nenhuma amostra repetida, o que
comprova que cada leitura corresponde a uma atualização nova do registrador do
sensor, e não a uma releitura do mesmo valor.

### Observação registrada durante o ensaio
Com o celular vibrando, **todas as três classes passaram a ser preditas**. Isso
não é evidência de funcionamento do classificador — pelo contrário. Duas janelas
estatisticamente indistinguíveis produziram classes opostas com 99,6% de
confiança:

| | Janela 242 | Janela 250 |
|---|---|---|
| desvio padrão da entrada | 0,148547 g | 0,150548 g |
| razão sobre a amplitude de treino | 5,955x | 6,036x |
| níveis int8 distintos | 110 | 109 |
| amostras ceifadas na quantização | 366 | 378 |
| **classe predita** | **outer_race (99,6%)** | **inner_race (99,6%)** |

A variável que separou as classes foi a severidade do ceifamento no quantizador
de entrada, não a vibração. O quantizador foi calibrado sobre o CWRU
(`scale = 0,02675884`); com a entrada a ~6x a amplitude de treino, cerca de 380
das 512 amostras ficam presas em ±127 e o modelo recebe uma onda quadrada.

O firmware passou a emitir alerta explícito nessa condição:

```
[!!] Quantizador ceifou 383 de 512 amostras (75%). O sinal chega ACHATADO
     ao modelo - a classe predita nao tem valor.
[!!] Amplitude 6.5x a do treino: o modelo opera fora da faixa que conhece.
```

**Conclusão de V1:** aquisição aprovada; classificação sob excitação de mesa é
inválida por construção e deve ser apresentada como tal.

---

## 3. Ensaio V2 — Inferência embarcada (comando `t`)

### Objetivo
Comprovar que o caminho de inferência dentro do ESP32 reproduz o do PC. Sem esse
ensaio, a acurácia do relatório é uma propriedade do notebook, não do produto
embarcado.

### Como funciona
O `build.py` grava na flash do ESP32 as **mesmas janelas do conjunto de teste**
que produziram a acurácia do relatório, junto com a saída de referência do
interpretador TFLite rodando no PC. O comando `t` reprocessa essas janelas dentro
do microcontrolador e compara.

Decisões de projeto relevantes:

- **As janelas são exportadas antes da normalização**, em unidades físicas. O
  ESP32 aplica a própria normalização e a própria quantização, de modo que o
  ensaio exercita a cadeia inteira e não apenas o `Invoke()`.
- **Armazenamento em `int16`** (~15 bits úteis) em vez de `float32`. O
  quantizador de entrada preserva 8 bits, então nada que o modelo enxergue se
  perde, e o custo em flash cai pela metade. É também o formato nativo do
  MPU6050.
- **A referência guarda o tensor de saída `int8` completo**, não só a classe
  vencedora. Comparar os bytes permite afirmar identidade aritmética entre PC e
  ESP32 — alegação bem mais forte do que concordância de `argmax`.
- **A quantização em Python usa arredondamento que afasta do zero**
  (`round_half_away`), replicando o `lroundf()` da libc do ESP32. O `np.round()`
  usa arredondamento bancário e produziria divergências de 1 LSB inexistentes.
- **A remoção de DC fica desligada** durante o ensaio. O CWRU já é um sinal AC de
  média ≈ 0; a remoção existe no laço principal apenas para descontar a gravidade
  lida pelo MPU6050, que não tem equivalente no dataset.

### Procedimento
1. `python build.py` — a etapa `[5/6]` gera `include/test_vectors.h` e
   `src/test_vectors.cpp`.
2. `cd arduino_deploy/tinyml_esp32 && pio run -t upload`.
3. No monitor serial, digitar `t`.
4. Copiar a saída inteira para o anexo do TCC.

O ensaio é disparado sob demanda e não interfere na operação normal: ao terminar,
o firmware volta a amostrar o sensor no ciclo padrão.

### Critérios de aceitação

| Métrica | Critério | Significado se falhar |
|---|---|---|
| **Entrada int8 idêntica ao PC** | 100% | Erro na normalização ou na quantização |
| **Mesma classe que o PC** | ≥ 98% | Divergência que altera decisões |
| Acurácia no ESP32 | ≈ acurácia da referência | Vetor ou constantes divergentes |
| `Invoke()` falhou | 0 janelas | Arena insuficiente ou operador ausente |
| Saída int8 idêntica | *diagnóstico, não critério* | ver §3.1 |

### Registro do ensaio — PC (13/08/2026)

| Métrica | Valor |
|---|---|
| Janelas gravadas | 112 de 112 (16 / 48 / 48 por classe) |
| Escala do vetor | 2,928110689e-06 g/LSB |
| Custo em flash | 112,9 KB |
| Acurácia do modelo Keras (float32) | 97,32% |
| Acurácia da referência TFLite int8 (kernels `BUILTIN_REF`) | 96,43% |
| Acurácia com kernels otimizados de desktop (`BUILTIN`) | 96,43% |
| Divergência entre os dois conjuntos de kernels no PC | 0 LSB, 100% idênticas |

A diferença de 0,89 pp entre Keras e TFLite corresponde a **exatamente uma
janela** (109 contra 108 acertos em 112). Esse é o custo medido da quantização
int8 e deve constar do relatório — é um resultado, não um defeito.

A referência gravada usa os kernels `BUILTIN_REF`, que são os que o TensorFlow
Lite **Micro** implementa. Os kernels otimizados de desktop (XNNPACK/ruy) foram
verificados e produzem saída idêntica, então essa escolha não afeta os números —
mas mantém a comparação metodologicamente correta.

### Registro do ensaio — ESP32 (13/08/2026)

| Métrica | Valor |
|---|---|
| **Entrada int8 idêntica ao PC** | **100,00%** (112/112) |
| Acurácia no ESP32 | 98,21% (110/112) |
| Mesma classe que o PC | **98,21%** (110/112) |
| Saída int8 idêntica ao PC | 75,00% (84/112), desvio máx. 22 LSB |
| ↳ descontando as saturadas | 41,67% (20/48) |
| Saídas saturadas (`int8 = 127`) | 57,14% (64/112) |
| `Invoke()` falhou | 0 janelas |
| Inferência (mín / médio / máx) | 9.307 / 9.392 / 9.856 µs |
| Vazão máxima | 106,5 janelas/s |
| Ciclo útil (janela de 512 ms) | 1,83% |
| Arena utilizada | 4.716 de 24.576 bytes (19,2%) |

Matriz de confusão medida no hardware:

| verdade \ predito | normal | inner_race | outer_race | recall |
|---|---|---|---|---|
| `0_normal` | 16 | 0 | 0 | 100,00% |
| `1_inner_race` | 0 | 48 | 0 | 100,00% |
| `2_outer_race` | 0 | 2 | 46 | 95,83% |

### 3.1 Interpretação da divergência de 25%

A taxa de igualdade byte a byte (75%) exige leitura cuidadosa, e três fatos a
explicam:

**1. As janelas "idênticas" incluem saturação trivial.** As 64 janelas das
classes 0 e 1 saíram todas com confiança 0,9961 — que é `int8 = 127`, o teto da
representação. Nessa condição qualquer diferença aritmética interna é ceifada
pelo limite superior, e a igualdade é trivial. Elas não constituem evidência de
identidade aritmética. O firmware passou a reportar essa fração separadamente.

**2. Toda a divergência está na classe 2.** As 28 janelas com desvio diferente de
zero são exatamente as de `2_outer_race`, a única classe cujas confianças variam
(0,55 a 0,99) em vez de saturar.

**3. O desvio cresce onde a confiança cai.** Correlação direta nos dados:

| Janelas | Confiança | Desvio |
|---|---|---|
| #078–#092 | 0,98–0,99 | 0 LSB |
| #101–#105 | 0,80–0,86 | 6–18 LSB |
| #107–#111 | 0,55–0,77 | 9–22 LSB |

Esse é o comportamento esperado da softmax: uma diferença de 1–2 LSB nos logitos
produz variação grande na probabilidade quando as classes estão empatadas, e
quase nenhuma quando uma delas domina. **Divergência grande com confiança alta
seria o sinal preocupante; o padrão observado é o inverso.**

**4. A entrada bate em 100%.** O firmware compara o *hash FNV-1a da janela já
quantizada* contra o valor calculado no PC, e o resultado foi **112/112**. Isso
prova, byte a byte, que:

- o armazenamento em `int16` e a reconstrução para `float` são exatos;
- a normalização `(x − 0,0182876) / 0,0249429` produz o mesmo resultado nas duas
  pontas, em `float32`;
- a quantização — incluindo a equivalência entre `lroundf()` e `round_half_away`
  — é idêntica.

Com a entrada provada igual e a saída divergente, **a causa está isolada por
eliminação nos kernels**: TensorFlow Lite Micro e TensorFlow Lite são
implementações distintas da mesma especificação int8.

#### Confirmação quantitativa

A afirmação "a softmax amplifica 1–2 LSB" foi medida, não apenas alegada. O
script `tools/sensibilidade_softmax.py` perturba cada logito em ±1 LSB
(escala = 0,2015) e mede o deslocamento da saída quantizada:

| Faixa de confiança | Janelas | Desvio médio | Desvio máx. |
|---|---|---|---|
| 0,00 – 0,70 | 4 | 12,5 LSB | 13 LSB |
| 0,70 – 0,90 | 13 | 8,5 LSB | 11 LSB |
| 0,90 – 0,99 | 22 | 2,4 LSB | 5 LSB |
| 0,99 – 1,00 | 73 | 0,1 LSB | 1 LSB |

Comparando com o que o ESP32 realmente produziu:

| Faixa | Previsto para 1 LSB | Observado no ESP32 |
|---|---|---|
| ≥ 0,99 | máx. 1 | 0 a 1 ✅ |
| 0,90 – 0,99 | máx. 5 | 0 a 5 ✅ |
| 0,70 – 0,90 | máx. 11 | 6 a 20 (≈ 2 LSB) |
| < 0,70 | máx. 13 | 12 a 22 (≈ 2 LSB) |

As duas primeiras faixas são explicadas exatamente por **1 LSB** de diferença nos
logitos; as duas últimas, por **2 LSB**. A sensibilidade é cerca de 100x maior nas
janelas de baixa confiança do que nas saturadas.

**Conclusão:** a divergência entre TFLM e TFLite está limitada a ≈ 2 LSB nos
logitos — cerca de 0,4 em unidades de logito — o que é o esperado para
implementações independentes de acumulação e requantização em int8. Não há
defeito no deploy.

### 3.2 O ESP32 acertou duas janelas que o PC errou

| Janela | Verdade | PC | ESP32 | Confiança |
|---|---|---|---|---|
| #108 | 2 | 1 ✗ | 2 ✓ | 0,5508 |
| #109 | 2 | 1 ✗ | 1 ✗ | 0,6484 |
| #110 | 2 | 1 ✗ | 2 ✓ | 0,5508 |
| #111 | 2 | 1 ✗ | 1 ✗ | 0,7695 |

O PC erra quatro janelas (108/112 = 96,43%); o ESP32 erra duas (110/112 =
98,21%). **Isso não é uma melhoria.** As quatro janelas estão na fronteira de
decisão, com confiança entre 0,55 e 0,77, e são justamente as de maior desvio em
LSB. O microcontrolador caiu do lado certo em duas delas por ruído numérico, não
por mérito.

**A acurácia a declarar no TCC é 96,43%**, a da referência. Reportar 98,21%
seria selecionar o número favorável produzido por ruído de arredondamento.

---

## 4. Custo do ensaio V2 no hardware

Medido com `pio run`, comparando o firmware antes e depois da inclusão dos
vetores:

| Recurso | Antes | Depois | Δ |
|---|---|---|---|
| Flash | 385.001 B (29,4%) | 503.537 B (38,4%) | +115 KB |
| RAM | 52.224 B (15,9%) | 52.256 B (15,9%) | +32 B |

Os vetores residem em `.rodata`, mapeada na flash (DROM) do ESP32, e não
consomem SRAM. Os 32 bytes adicionais de RAM são a matriz de confusão.

Para reduzir o custo, ajustar `MAX_TEST_VECTORS` no `build.py`; a seleção passa a
ser estratificada e preserva a proporção entre as classes. Para eliminá-lo,
apagar `include/test_vectors.h` e `src/test_vectors.cpp` — o firmware detecta a
ausência via `__has_include` e continua compilando, com o comando `t` informando
que o conjunto não foi gerado.

---

## 5. Ensaio V3 — Bancada rotativa (não executado)

Para validar a classificação de falhas seria necessário:

- motor com eixo, rolamento montado e carga radial;
- pelo menos um rolamento com defeito na pista externa e um na pista interna;
- rotação estável, medida com tacômetro, para calcular BPFI e BPFO esperados;
- acelerômetro fixado rigidamente ao mancal, não à protoboard.

Enquanto V3 não existir, a afirmação defensável do trabalho é:

> O sistema adquire vibração a 1 kHz com jitter inferior a 1,4% do intervalo de
> amostragem e executa a inferência da CNN quantizada em int8 no ESP32 em
> 9,4 ms, ocupando 4,7 KB de arena e 1,8% do ciclo de aquisição. A inferência
> embarcada reproduz a decisão do interpretador de referência em 110 das 112
> janelas de teste; as duas divergências ocorrem em janelas de baixa confiança
> (< 0,78), onde a softmax amplifica diferenças de arredondamento entre as
> implementações do TensorFlow Lite e do TensorFlow Lite Micro. A acurácia de
> 96,43% refere-se ao conjunto de teste do dataset CWRU.

E não:

> O sistema detecta falhas de rolamento com 96,43% de acurácia.

---

## 6. Limitações a declarar no relatório

1. **Volume de dados.** Após a decimação para 1 kHz existem apenas **49 janelas
   independentes** (25,4 s de sinal ÷ 512 amostras). As 112 janelas de teste
   têm sobreposição de 93,75% e não são amostras estatisticamente independentes.
2. **Seleção de época no conjunto de teste.** Não há dados suficientes para um
   terceiro conjunto, então o `EarlyStopping` usa o próprio conjunto de teste
   para escolher a melhor época. A acurácia é, portanto, **otimista**.
3. **Banda de frequência.** O CWRU foi gravado a 12 kHz (48 kHz na baseline); o
   MPU6050 não passa de 1 kHz. A decimação para 1 kHz descarta o conteúdo acima
   de 500 Hz, incluindo a banda de ressonância de 2–5 kHz classicamente usada em
   diagnóstico de rolamentos. O modelo opera apenas sobre as componentes de baixa
   frequência.
4. **Faixa de amplitude.** O quantizador de entrada foi calibrado sobre o CWRU.
   Sinais fora da faixa de aproximadamente 0,1x a 2,0x a amplitude de treino são
   ceifados e produzem classificação sem significado. O firmware alerta nessa
   condição.

Mitigação disponível para o item 1: baixar mais arquivos do CWRU com
`download_cwru.py` e reexecutar o `build.py`.

---

## 7. Referência rápida dos comandos do console

| Tecla | Ação |
|---|---|
| `h` | ajuda |
| `s` | reexecuta o diagnóstico do sensor |
| `d` | liga/desliga o modo verboso (amostras cruas) |
| `x` `y` `z` `m` | seleciona o eixo enviado ao modelo (`m` = módulo) |
| `o` | liga/desliga a remoção do nível DC (gravidade) |
| `f` | alterna fs: 250 → 500 → 1000 → 2000 Hz |
| `r` | despeja a janela atual em CSV, para FFT |
| `t` | **executa a validação embarcada com o conjunto de teste** |
| `p` | pausa/retoma a inferência |

---

## 8. Arquivos do protocolo

| Caminho | Papel |
|---|---|
| `build.py`, etapa `[5/6]` | Exporta o conjunto de teste e a saída de referência |
| `src/pipeline/c_generator.py` | Gera `test_vectors.h` / `test_vectors.cpp` |
| `arduino_deploy/.../src/main.cpp` | `runTestVectorSuite()` — o comando `t` |
| `tools/sensibilidade_softmax.py` | Mede a sensibilidade da softmax a 1 LSB (§3.1) |
| `relatorios/relatorio_metricas.md` | Métricas do modelo no PC |
| `docs/protocolo_validacao.md` | Este documento |

O pipeline é determinístico (`SEED = 42` aplicado a NumPy, `random` e
TensorFlow): reexecutar `python build.py` sobre os mesmos dados regenera
artefatos byte a byte idênticos, verificado via `git status`.
