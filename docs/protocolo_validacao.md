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
| **V4** | Generalização entre condições de operação | Validação deixando uma carga de fora | ⚠️ aprovado com ressalva |

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
| Divergência de classe | apenas em janelas de confiança < 0,90 | ver §3.1 |

### Registro do ensaio — PC, dataset atual de 28 arquivos (13/08/2026)

| Métrica | Valor |
|---|---|
| Janelas gravadas na flash | 128 de 1.313 (16 / 56 / 56 por classe) |
| Escala do vetor | 6,626966297e-06 g/LSB |
| Custo em flash | 129,0 KB |
| Acurácia Keras float32, conjunto de teste completo | 99,39% |
| Acurácia TFLite int8, conjunto de teste completo | 99,77% |
| Acurácia da referência sobre as 128 janelas exportadas | 99,22% |
| Divergência entre `BUILTIN` e `BUILTIN_REF` no PC | 0 LSB, 100% idênticas |

As três acurácias medem coisas diferentes e não devem ser comparadas entre si: as
duas primeiras cobrem as 1.313 janelas de teste; a terceira, apenas o subconjunto
de 128 gravado no microcontrolador.

A referência usa os kernels `BUILTIN_REF`, que são os que o TensorFlow Lite
**Micro** implementa. Os otimizados de desktop (XNNPACK/ruy) foram verificados e
produzem saída idêntica, então a escolha não afeta os números — mas mantém a
comparação metodologicamente correta.

### Registro do ensaio — ESP32, dataset de 28 arquivos (13/08/2026)

| Métrica | Valor |
|---|---|
| **Entrada int8 idêntica ao PC** | **100,00%** (128/128) |
| **Mesma classe que o PC** | **99,22%** (127/128) |
| Acurácia no ESP32 | 98,44% (126/128) |
| Acurácia da referência sobre as mesmas 128 janelas | 99,22% (127/128) |
| Saída int8 idêntica ao PC | 89,84% (115/128), desvio máx. 104 LSB |
| Saídas com a classe vencedora saturada (`int8 = 127`) | 94,53% (121/128) |
| `Invoke()` falhou | 0 janelas |
| Inferência (mín / médio / máx) | 9.207 / 9.299 / 9.697 µs |
| Vazão máxima | 107,5 janelas/s |
| Ciclo útil (janela de 512 ms) | 1,82% |
| Arena utilizada | 4.716 de 24.576 bytes (19,2%) |

Matriz de confusão medida no hardware:

| verdade \ predito | normal | inner_race | outer_race | recall |
|---|---|---|---|---|
| `0_normal` | 16 | 0 | 0 | 100,00% |
| `1_inner_race` | 0 | 54 | 2 | 96,43% |
| `2_outer_race` | 0 | 0 | 56 | 100,00% |

**Veredito: aprovado.** A entrada é bit a bit idêntica à do PC, e das 128 janelas
apenas uma (#035) recebe classe diferente da referência. Nenhuma falha de
`Invoke()`, arena em 19% e ciclo útil de 1,8%.

### 3.1 Onde ficam as divergências

Toda a divergência se concentra em **7 janelas** — as únicas em que a softmax não
satura. As outras 121 têm a classe vencedora em `int8 = 127`.

| Janela | Verdade | Confiança | Desvio | Observação |
|---|---|---|---|---|
| #034 | 1 | 0,5000 | 52 LSB | empate exato entre duas classes |
| #035 | 1 | 0,7031 | **104 LSB** | única divergência de classe |
| #036 | 1 | 0,5000 | 52 LSB | empate exato |
| #037 | 1 | 0,8477 | 37 LSB | PC e ESP32 erram juntos |
| #038 | 1 | 0,5000 | 52 LSB | empate exato |
| #074 | 2 | 0,9688 | 10 LSB | |
| #098 | 2 | 0,9688 | 7 LSB | |

Confiança de 0,5000 é `int8 = 0`: um empate perfeito entre duas classes. As
janelas #034 a #038 são um trecho contíguo de sinal em que o modelo genuinamente
não decide.

#### Confirmação quantitativa

O script `tools/sensibilidade_softmax.py` perturba cada logito em ±1 LSB
(escala = 0,8608 neste modelo) e mede o deslocamento da saída quantizada:

| Faixa de confiança | Janelas | Desvio médio | Desvio máx. |
|---|---|---|---|
| 0,00 – 0,70 | 16 | 52,0 LSB | 52 LSB |
| 0,70 – 0,90 | 33 | 51,5 LSB | 52 LSB |
| 0,90 – 0,99 | 45 | 9,7 LSB | 21 LSB |
| 0,99 – 1,00 | 1.219 | 0,1 LSB | 2 LSB |

O casamento com o hardware é exato:

- #034, #036 e #038 têm desvio de **52 LSB — precisamente o previsto para 1 LSB**
  de diferença nos logitos.
- #035 tem **104 LSB = 2 × 52**, ou seja, 2 LSB.
- #074 e #098, com 10 e 7 LSB, caem na banda 0,90–0,99, cuja média prevista é 9,7.

**A divergência entre TFLite Micro e TFLite continua limitada a 1–2 LSB nos
logitos**, o mesmo resultado obtido com o dataset anterior. Os desvios em LSB da
saída cresceram (104 contra 22) apenas porque a escala dos logitos subiu de
0,2015 para 0,8608 no modelo retreinado — 4,3x. Não é perda de fidelidade: é o
mesmo erro aritmético visto através de um quantizador mais grosseiro.

#### Ressalva sobre a métrica "descontando as saturadas"

O firmware reporta `0,00% (0/7)` nessa linha. O denominador de 7 torna a
porcentagem inútil, e o rótulo é impreciso: a saturação garante igualdade apenas
no byte da **classe vencedora**. Seis janelas saturadas (#073, #091, #092, #093,
#094, #097) têm desvio de 1 a 7 LSB nas classes perdedoras.

Com o dataset maior o modelo ficou muito mais confiante — 94,5% de saídas
saturadas contra 57,1% antes — e a amostra não saturada encolheu a ponto de a
métrica perder sentido. **As métricas que decidem o deploy são as duas primeiras
da tabela: entrada idêntica (100%) e mesma classe que o PC (99,22%).**

### 3.2 O sentido da diferença é aleatório

| Execução | Acurácia ESP32 | Acurácia PC | Diferença |
|---|---|---|---|
| Dataset de 3 arquivos | 98,21% (110/112) | 96,43% (108/112) | ESP32 **+2** janelas |
| Dataset de 28 arquivos | 98,44% (126/128) | 99,22% (127/128) | ESP32 **−1** janela |

Na primeira execução o microcontrolador acertou duas janelas que o PC errou; na
segunda, errou uma que o PC acertou. Isso encerra qualquer tentação de tratar a
diferença como mérito ou defeito: **é ruído de arredondamento em janelas de
fronteira, e muda de sinal entre execuções.**

Reporte a acurácia da referência, não a do microcontrolador.


## 4. Custo do ensaio V2 no hardware

Medido com `pio run`, comparando o firmware antes e depois da inclusão dos
vetores:

| Recurso | Antes | Depois | Δ |
|---|---|---|---|
| Flash | 385.001 B (29,4%) | 520.801 B (39,7%) | +132 KB |
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
> embarcada reproduz a decisão do interpretador de referência, e as divergências
> residuais se restringem a janelas de baixa confiança, onde a softmax amplifica
> diferenças de arredondamento entre as implementações do TensorFlow Lite e do
> TensorFlow Lite Micro. Sobre o conjunto de teste do CWRU, a acurácia é de
> 99,77% com divisão temporal intra-arquivo e de 86,01% quando o modelo é
> avaliado numa condição de carga inédita — 99,50% se excluída a condição sem
> carga, na qual o método não é confiável (§6).

E não:

> O sistema detecta falhas de rolamento com 99,77% de acurácia.

---

## 6. Ensaio V4 — Generalização entre condições de operação

### O problema
O `build.py` divide treino e teste por trecho temporal **dentro de cada
arquivo**. Não há vazamento de amostras — existe uma zona morta de uma janela
inteira na fronteira — mas toda janela de teste tem uma contraparte de treino da
**mesma gravação**: mesmo rolamento, mesmo defeito, mesma carga, mesma montagem,
mesma sessão. É vazamento no nível da gravação, e é o que sustenta a acurácia de
99,77%.

### Procedimento
`python tools/validacao_por_carga.py`. Treina com os primeiros 85% dos arquivos
de três níveis de carga (validação nos 15% finais, só para o *early stopping*) e
testa em **todos** os arquivos da quarta carga, em janelas sem sobreposição.

### Resultado medido (13/08/2026)

| Teste | rpm | Janelas | Acurácia | Detecção | Falso alarme | Recall normal / inner / outer |
|---|---|---|---|---|---|---|
| 0 hp | 1797 | 123 | **45,53%** | **42,98%** | 0,00% | 100,0% / 47,4% / 35,1% |
| 1 hp | 1772 | 133 | 100,00% | 100,00% | 0,00% | 100,0% / 100,0% / 100,0% |
| 2 hp | 1750 | 133 | 100,00% | 100,00% | 0,00% | 100,0% / 100,0% / 100,0% |
| 3 hp | 1730 | 134 | 98,51% | 100,00% | 0,00% | 100,0% / 96,6% / 100,0% |
| **Média** | | | **86,01%** (σ 23,38 pp) | **85,75%** (σ 24,69 pp) | 0,00% | |

*Detecção* = fração das janelas com falha que **não** foram chamadas de normal.

### Interpretação

**O modelo generaliza entre cargas — exceto a 0 hp.** Com o rolamento sob carga
(1 a 3 hp), a acurácia numa condição inédita fica entre 98,5% e 100%. Sem carga,
desaba para 45,5%.

**E o modo de falha é o pior possível.** A 0 hp, a detecção cai para 42,98%: mais
da metade das janelas de rolamento defeituoso é classificada como **normal**. Não
é confusão entre pista interna e externa — é falha de detecção. Num sistema de
manutenção preditiva, esse é o erro caro.

A explicação é física e coerente: sem carga radial, os elementos rolantes batem
no defeito com pouca energia, e a assinatura de impacto se aproxima da linha de
base saudável. A decimação para 1 kHz agrava o quadro, porque descarta a banda de
ressonância de 2–5 kHz onde esses impactos fracos ainda seriam visíveis.

O **falso alarme é 0% em todas as cargas**: o modelo nunca acusa defeito num
rolamento saudável. Ele é conservador — e para manutenção preditiva esse é
justamente o viés indesejado, porque o custo de uma parada não detectada supera o
de uma inspeção desnecessária.

### Consequência para o TCC

Reporte os dois números, com os papéis explícitos:

| Protocolo | Acurácia | O que mede |
|---|---|---|
| Divisão temporal intra-arquivo | 99,77% | limite superior; consistência dentro da condição |
| Deixando uma carga de fora | 86,01% | generalização para condição inédita |
| ↳ apenas cargas 1–3 hp | 99,50% | generalização sob carga |

E declare a limitação de operação: **o sistema não é confiável com o rolamento
descarregado**.

---

## 7. Limitações a declarar no relatório

1. **Generalização a 0 hp.** Ver §6. É a limitação mais séria do trabalho.
2. **Banda de frequência.** O CWRU foi gravado a 12 kHz (48 kHz na baseline); o
   MPU6050 não passa de 1 kHz. A decimação descarta o conteúdo acima de 500 Hz,
   incluindo a banda de ressonância de 2–5 kHz classicamente usada em
   diagnóstico de rolamentos. Essa é a causa provável da falha a 0 hp, e é uma
   limitação do sensor, não do modelo.
3. **Seleção de época no conjunto de teste.** No `build.py`, o `EarlyStopping`
   usa o próprio conjunto de teste para escolher a melhor época, o que torna a
   acurácia otimista. O `tools/validacao_por_carga.py` **não** tem esse defeito:
   valida num trecho separado das cargas de treino.
4. **Faixa de amplitude.** O quantizador de entrada foi calibrado sobre o CWRU.
   Sinais fora da faixa de aproximadamente 0,1x a 2,0x a amplitude de treino são
   ceifados e produzem classificação sem significado. O firmware alerta nessa
   condição.
5. **Diversidade de defeitos.** Foram usados apenas defeitos usinados por
   eletroerosão na pista, em três diâmetros. Falhas reais evoluem de forma
   progressiva e podem ter assinatura distinta.

### Volume de dados — resolvido

O conjunto passou de 3 para **28 arquivos** (`download_cwru.py`), cobrindo 4
níveis de carga e 3 diâmetros de defeito por classe de falha:

| | Antes | Depois |
|---|---|---|
| Arquivos | 3 | 28 |
| Sinal após decimação | 25,4 s | 279,4 s |
| **Janelas independentes** | **49** | **545** |
| Janelas de teste | 112 | 1.313 |

---

## 8. Referência rápida dos comandos do console

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

## 9. Arquivos do protocolo

| Caminho | Papel |
|---|---|
| `build.py`, etapa `[5/6]` | Exporta o conjunto de teste e a saída de referência |
| `src/pipeline/c_generator.py` | Gera `test_vectors.h` / `test_vectors.cpp` |
| `arduino_deploy/.../src/main.cpp` | `runTestVectorSuite()` — o comando `t` |
| `tools/sensibilidade_softmax.py` | Mede a sensibilidade da softmax a 1 LSB (§3.1) |
| `tools/validacao_por_carga.py` | Validação deixando uma carga de fora (§6) |
| `download_cwru.py` | Baixa os 28 arquivos do dataset |
| `relatorios/relatorio_metricas.md` | Métricas do modelo no PC |
| `docs/pipeline.md` | Como a pipeline funciona, etapa por etapa |
| `docs/protocolo_validacao.md` | Este documento |

O pipeline é determinístico (`SEED = 42` aplicado a NumPy, `random` e
TensorFlow): reexecutar `python build.py` sobre os mesmos dados regenera
artefatos byte a byte idênticos, verificado via `git status`.
