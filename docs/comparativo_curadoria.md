# Comparativo: dataset bruto x dataset curado

Documento de resultado. Confronta o sistema treinado com o dataset **como ele
vinha do CWRU** contra o sistema treinado com o dataset **depois da curadoria**,
nos três níveis em que o projeto mede desempenho.

| Nível | O que mede | Onde roda |
|---|---|---|
| **N1 · Sonda linear** | separabilidade do dado, sem envolver a CNN | `dataset_engineering` |
| **N2 · CNN no PC** | o modelo, float32 e int8 | `build.py` |
| **N3 · Deploy no ESP32** | a cadeia embarcada inteira | comando `t` no console serial |

> ### Estado de preenchimento
>
> A coluna **sem curadoria** está completa: os artefatos estão arquivados em
> [`../relatorios/sem_curadoria/`](../relatorios/sem_curadoria/).
>
> A coluna **com curadoria** tem apenas N1. N2 e N3 exigem reexecutar o
> `build.py` e regravar o ESP32 — ver [§5](#5-como-preencher-a-coluna-que-falta).

---

## 1. O que mudou no dataset

| | Sem curadoria | Com curadoria |
|---|---|---|
| Ensaios | 28 | 24 |
| Sinal a 1 kHz | 279,4 s | 238,7 s |
| Janelas independentes | 545 | 466 |
| Janelas de treino / teste | 6.104 / 1.313 | 5.214 / 1.121 |
| Teste por classe (normal / interna / externa) | 160 / 577 / 576 | 160 / 577 / 384 |
| Ensaios lendo a série de outro ensaio | 1 | 0 |
| Ensaios de falha sem assinatura observável | 4 | 0 |

Tratativas aplicadas: **T1** (canal selecionado pelo número do arquivo),
**T2** (inventário obrigatório), **T3** (deduplicação) e **T9** (quarentena por
observabilidade medida). Detalhamento em
[`../dataset_engineering/README.md`](../dataset_engineering/README.md).

> **A distribuição do teste por classe identifica o build.** É por ela que se
> confere se uma captura do comando `t` veio do firmware certo: a seleção dos
> 128 vetores gravados na flash é estratificada, então 1.313 janelas produzem
> **16 / 56 / 56** e 1.121 produzem **18 / 66 / 44**. Uma captura com a
> distribuição errada é de um deploy antigo, independentemente do que o resto
> do console diga.

---

## 2. N1 — Sonda linear (separabilidade do dado)

Regressão logística sobre atributos rasos, validada deixando uma carga de fora,
janelas independentes dos dois lados. Não é o modelo do projeto: é o termômetro
que isola a qualidade do **dado** do mérito da **arquitetura**.

| Medida | Sem curadoria | Com curadoria | Δ |
|---|---|---|---|
| Acurácia média | 84,02% | **89,04%** | **+5,02 pp** |
| Desvio entre cargas | 9,25 pp | **8,02 pp** | −1,23 pp |
| Detecção de falha | 95,61% | 94,47% | −1,14 pp |
| Falso alarme | 28,95% | **25,00%** | −3,95 pp |

### Por carga deixada de fora

| Conjunto | 0 hp | 1 hp | 2 hp | 3 hp |
|---|---|---|---|---|
| Sem curadoria | 69,11% | 83,46% | 91,73% | 91,79% |
| Com curadoria | **79,81%** | 82,46% | **98,25%** | **95,65%** |

O ganho se concentra em **0 hp (+10,7 pp)** e **2 hp (+6,5 pp)** — exatamente as
duas condições que a auditoria previa:

- **0 hp** era a condição mais frágil, e quatro dos ensaios removidos por T9
  incluíam justamente o `197.mat` (0,014", 0 hp).
- **2 hp** melhorou porque T1 recuperou o baseline real de 2 hp. Antes,
  `99.mat` entregava a série de `98.mat`, e a rodada de 2 hp testava em dado
  que já estava no treino — o número anterior era inflado por vazamento e, ao
  mesmo tempo, o modelo nunca via um rolamento saudável a 1.750 rpm.

A ablação separa as contribuições:

| Conjunto | Ensaios | Acurácia |
|---|---|---|
| bruto | 28 | 84,02% |
| + T1 | 28 | 84,58% |
| + T1 + T9 | 24 | **89,04%** |

T1 sozinho quase não move a sonda. Isso **não** o torna dispensável: ele corrige
a *correção* dos rótulos e fecha um vazamento treino/teste. Uma métrica que não
melhora depois de remover um vazamento é o resultado esperado — o vazamento
inflava o número, não o dado.

---

## 3. N2 — CNN no PC

Divisão temporal intra-arquivo: primeiros 80% de cada gravação para treino,
últimos 20% para teste, com zona morta de uma janela inteira na fronteira.

| Medida | Sem curadoria | Com curadoria |
|---|---|---|
| Keras float32 | 99,39% | *pendente* |
| **TFLite int8** (o que vai ao ESP32) | **99,77%** | *pendente* |
| Custo da quantização | +0,38 pp (5 de 1.313 janelas) | *pendente* |
| Flash | 15,24 KB | *pendente* |

Fonte: [`../relatorios/sem_curadoria/relatorio_metricas.md`](../relatorios/sem_curadoria/relatorio_metricas.md).

> **Este número é o limite superior, não o realista.** Toda janela de teste tem
> uma contraparte de treino da mesma gravação: mesmo rolamento, mesmo defeito,
> mesma carga, mesma montagem. Ele mede consistência dentro da condição de
> operação, não generalização.
>
> **Espere que ele caia com a curadoria, e isso é bom.** Os quatro ensaios em
> quarentena não tinham assinatura de defeito; o modelo só podia acertá-los
> decorando a gravação. Removê-los tira do numerador acertos que não eram
> diagnóstico. Um 99,77% que vira 99,3% com quatro ensaios impossíveis a menos
> é um modelo mais honesto, não pior.

### Generalização entre condições (leave-one-load-out com a CNN)

| Medida | Sem curadoria | Com curadoria |
|---|---|---|
| Acurácia média | 86,01% | *pendente* |
| ↳ apenas cargas 1–3 hp | 99,50% | *pendente* |
| Detecção a 0 hp | **43%** | *pendente* |

O colapso a 0 hp é o resultado mais importante do trabalho até aqui, e o que a
curadoria tem mais chance de mover. A sonda linear já indicou +10,7 pp nessa
carga.

---

## 4. N3 — Deploy no ESP32

Comando `t`: o ESP32 reprocessa as mesmas janelas de teste que geraram a
acurácia do relatório e compara com o interpretador TFLite do PC.

| Medida | Sem curadoria | Com curadoria |
|---|---|---|
| Distribuição das 128 janelas | 16 / 56 / 56 | 18 / 66 / 44 |
| **Entrada int8 idêntica ao PC** | **100,00%** (128/128) | *pendente* |
| **Mesma classe que o PC** | **99,22%** (127/128) | *pendente* |
| Acurácia no ESP32 | 98,44% (126/128) | *pendente* |
| Saída int8 idêntica ao PC | 89,84% — desvio máx. 104 LSB | *pendente* |
| Saídas saturadas (int8 = 127) | 94,53% | *pendente* |
| Inferência (mín / médio / máx) | 9.208 / 9.299 / 9.700 µs | *pendente* |
| Ciclo útil (janela de 512 ms) | 1,82% | *pendente* |
| Tensor arena | 4.716 de 24.576 bytes | *pendente* |

Captura bruta: [`../relatorios/sem_curadoria/validacao_embarcada.txt`](../relatorios/sem_curadoria/validacao_embarcada.txt).

### Matriz de confusão no hardware — sem curadoria

| Verdade \ Predito | normal | interna | externa | Recall |
|---|---|---|---|---|
| **normal** | 16 | 0 | 0 | 100,00% |
| **interna** | 0 | 54 | 2 | 96,43% |
| **externa** | 0 | 0 | 56 | 100,00% |

### Leitura do resultado

**A cadeia de aquisição está provada.** `Entrada int8 idêntica ao PC = 100%`
significa que a janela que o ESP32 monta — depois de normalizar com a média e o
desvio do treino e quantizar para int8 — é **byte a byte** a mesma que o PC
usou, verificada por hash FNV-1a. Sem essa checagem, uma divergência na saída
seria ambígua entre normalização, quantização e kernels. Com ela, o erro fica
isolado na aritmética.

**As duas divergências são de naturezas diferentes:**

| Janela | Verdade | PC | ESP32 | Confiança | Desvio | Natureza |
|---|---|---|---|---|---|---|
| #035 | interna | interna | externa | 0,7031 | 104 LSB | divergência **do deploy** |
| #037 | interna | externa | externa | 0,8477 | 37 LSB | erro **do modelo** (o PC também erra) |

Só **#035** conta contra o embarcado — 1 em 128, ou 0,78%. **#037** o PC erra
igual: é limitação do modelo, e apareceria idêntica num servidor.

**O desvio em LSB cresce onde a confiança cai, que é o comportamento esperado.**
As janelas #034, #036 e #038 saem com confiança exatamente 0,5000 e desvio de
52 LSB, e ainda assim PC e ESP32 concordam na classe. São janelas em que duas
classes estão empatadas nos logitos; a softmax int8 amplifica diferenças de 1–2
LSB para dezenas de LSB na saída. O sinal preocupante seria o inverso —
divergência grande com confiança alta —, e ele não ocorre em nenhuma das 128.

**As 94,53% de saídas saturadas explicam por que "saída idêntica" é a métrica
errada para decidir o deploy.** Quando o logito satura em int8 = 127, a
igualdade é trivial e não prova nada sobre a aritmética. Descontando as
saturadas, nenhuma das 7 restantes bate exatamente — e mesmo assim a decisão de
classe coincide em 127 de 128. É por isso que **`mesma classe que o PC` é a
métrica que decide**, e não `saída int8 idêntica`.

**Margem de hardware confortável.** 9,3 ms de inferência numa janela de 512 ms
dá 1,82% de ciclo útil; a arena usa 4.716 dos 24.576 bytes reservados (19%).
Nenhum dos dois é gargalo, e a curadoria não deve mexer neles — o modelo tem a
mesma arquitetura e o mesmo número de parâmetros. Se esses números mudarem
muito no novo build, é sinal de que algo além do dataset mudou.

---

## 5. Como preencher a coluna que falta

```bash
# 1. Confirme que a curadoria está aplicada (deve dizer 24 ensaios)
python -m dataset_engineering.executar --somente 3

# 2. Retreine. A primeira linha deve dizer "Fonte: dataset curado v2..."
python build.py

# 3. Generalização entre cargas com a CNN
python tools/validacao_por_carga.py

# 4. Regrave o ESP32 — sem isto o comando 't' testa o modelo antigo
cd arduino_deploy/tinyml_esp32
pio run -t upload
pio device monitor
#    e então tecle 't'
```

**Verificação obrigatória antes de aceitar a captura do passo 4:** a matriz de
confusão precisa somar **18 / 66 / 44** nas linhas. Se somar 16 / 56 / 56, o
firmware não foi regravado e a captura é do deploy antigo.

O `relatorio_metricas.md` gerado pelo passo 2 agora traz a linha
**Fonte dos dados** e a distribuição do teste por classe, justamente para que
essa conferência não dependa de memória.

---

## Ver também

- [`dataset.md`](dataset.md) — a auditoria que motivou a curadoria
- [`../dataset_engineering/README.md`](../dataset_engineering/README.md) — o fluxo de curadoria
- [`../dataset_engineering/relatorios/engenharia_dados.md`](../dataset_engineering/relatorios/engenharia_dados.md) — o comparativo gerado do N1
- [`protocolo_validacao.md`](protocolo_validacao.md) — o protocolo dos ensaios V1–V4
