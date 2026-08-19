# O Dataset — Primeira Versão

Auditoria do conjunto de dados usado para treinar o modelo embarcado. Descreve
**o que existe em `data/`, o que dele chega ao ESP32 e o que se perde no
caminho**.

Todos os números deste documento são reproduzíveis:

```bash
python tools/analise_dataset.py
```

> ### Estado atual
>
> Este documento é o **registro da auditoria** — descreve o dataset como ele foi
> encontrado. Os defeitos apontados aqui já foram tratados pelo fluxo em
> [`../dataset_engineering/`](../dataset_engineering/README.md), que produz o
> dataset curado consumido pelo `build.py`.
>
> | Achado | Situação |
> |---|---|
> | 1 · Três arquivos de 0 hp ausentes | ✅ baixados; inventário 28/28 |
> | 2 · `99.mat` entregando o sinal de `98.mat` | ✅ corrigido (T1) |
> | 3 · Quatro ensaios sem assinatura observável | ✅ em quarentena (T9) |
> | 4 · Desbalanceamento estrutural de classes | ⚠️ mitigado por peso; teto é do CWRU |
> | 5 · Relatórios dessincronizados | ⏳ aguardando novo `build.py` |
>
> O resultado medido do sistema com e sem curadoria está em
> [`comparativo_curadoria.md`](comparativo_curadoria.md).
>
> O efeito medido da curadoria está em
> [`../dataset_engineering/relatorios/engenharia_dados.md`](../dataset_engineering/relatorios/engenharia_dados.md).

> **Por que este documento existe.** O resto do projeto trata o dataset como
> entrada dada. Mas o teto de desempenho do sistema não está na arquitetura da
> CNN — está aqui. A decimação de 12 kHz para 1 kHz, imposta pelo MPU6050,
> descarta **98% da energia dos sinais de falha**. Documentar essa perda é o que
> transforma "o modelo errou" em "o sensor não vê".

---

## 1. Origem e estrutura

| | |
|---|---|
| Fonte | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) |
| Rolamento | SKF 6205-2RS JEM, lado do acionamento (*drive end*) |
| Defeitos | usinados por eletroerosão (EDM) |
| Aquisição | acelerômetro no mancal, 12 kHz (falhas) / 48 kHz (baseline) |
| Duração por ensaio | ≈ 10 s |

Cada arquivo `.mat` é **um ensaio**: um rolamento específico, com um defeito de
diâmetro específico, sob uma carga específica. O nome do arquivo é o número de
catálogo do CWRU e não descreve nada por si — o mapeamento está em
`download_cwru.py` (`CATALOGO`) e em `tools/analise_dataset.py`
(`CARGA_POR_ARQUIVO`, `SEVERIDADE_POR_ARQUIVO`).

Dentro do `.mat` há até quatro variáveis: `X###_DE_time` (drive end),
`X###_FE_time` (fan end), `X###_BA_time` (base) e `X###RPM`. **O projeto usa
apenas o canal DE** — é o mais próximo do defeito e o que corresponde à posição
onde o MPU6050 seria montado.

### Frequências características

O defeito não produz um tom: produz uma sequência periódica de impactos, cuja
taxa depende da geometria do rolamento e da rotação do eixo.

| Defeito | Multiplicador | A 1797 rpm | A 1730 rpm |
|---|---|---|---|
| Pista interna (BPFI) | 5,4152 × f<sub>r</sub> | 162,2 Hz | 156,1 Hz |
| Pista externa (BPFO) | 3,5848 × f<sub>r</sub> | 107,4 Hz | 103,4 Hz |

Ambas ficam **abaixo do Nyquist de 500 Hz** — este é o fato que salva o projeto,
e a razão pela qual a Seção 5 conclui que a tarefa continua possível a 1 kHz.

---

## 2. Inventário da v1

O catálogo pede 28 arquivos. **Há 25 em disco.**

| Classe | Arquivos | Sinal a 1 kHz | % do total | Janelas independentes |
|---|---|---|---|---|
| `0_normal` | 3 | 30,3 s | 11,9% | 59 |
| `1_inner_race` | 11 | 111,8 s | 44,0% | 218 |
| `2_outer_race` | 11 | 111,9 s | 44,1% | 218 |
| **Total** | **25** | **254,0 s** | | **496** |

> "Janelas independentes" = sinal total ÷ 512 amostras. O `build.py` usa
> sobreposição de 93,75%, que produz 6.104 janelas de treino e 1.313 de teste —
> mas isso multiplica a contagem sem criar informação nova. **496 é o número a
> citar ao discutir validade estatística.**

### Cobertura do plano fatorial

| Classe | Severidade | 0 hp | 1 hp | 2 hp | 3 hp |
|---|---|---|---|---|---|
| `0_normal` | — | ❌ | ✅ | ⚠️ | ✅ |
| `1_inner_race` | 0,007" | ❌ | ✅ | ✅ | ✅ |
| `1_inner_race` | 0,014" | ✅ | ✅ | ✅ | ✅ |
| `1_inner_race` | 0,021" | ✅ | ✅ | ✅ | ✅ |
| `2_outer_race` | 0,007" | ❌ | ✅ | ✅ | ✅ |
| `2_outer_race` | 0,014" | ✅ | ✅ | ✅ | ✅ |
| `2_outer_race` | 0,021" | ✅ | ✅ | ✅ | ✅ |

❌ = ausente · ⚠️ = presente mas duplicado (ver Achado 2)

---

## 3. Achados da auditoria

### Achado 1 — Três arquivos ausentes, todos de 0 hp

`97.mat` (baseline 0 hp), `105.mat` e `130.mat` (0,007" a 0 hp) não estão em
`data/`. O `download_cwru.py` não interrompe a execução quando um download
falha; o `build.py` também não verifica o inventário — ele simplesmente treina
com o que encontrar.

**Consequência direta:** na validação `leave-one-load-out`, a rodada de 0 hp
testa um conjunto **sem nenhuma janela normal**. O recall da classe 0 e a taxa
de falso alarme são indefinidos nessa rodada, e o modelo nunca vê um rolamento
saudável descarregado. Isso agrava — mas não explica sozinho — a queda de
detecção para 43% em 0 hp registrada no `README.md`.

### Achado 2 — `99.mat` entrega o sinal de `98.mat`

O arquivo `99.mat` do CWRU contém, além das próprias variáveis, uma cópia das de
`98.mat`:

```
99.mat → {ans, X098_DE_time, X098_FE_time, X099_DE_time, X099_FE_time}
```

`DataProcessor.load_mat_file()` seleciona a **primeira** chave que contém
`_DE_time` (`src/pipeline/data_processor.py:27`). Nesse arquivo, a primeira é
`X098_DE_time`. Verificado amostra a amostra: as séries entregues por `98.mat` e
`99.mat` são **idênticas**.

**Consequências:**

1. A classe normal — já a minoritária — tem 3 arquivos mas apenas **2 gravações
   distintas**. O baseline de 2 hp nunca entra no treino.
2. Na validação por carga, `99.mat` é rotulado como 2 hp mas contém sinal de
   1 hp, que está no conjunto de **treino** daquela rodada. É vazamento direto
   de treino para teste, e infla a acurácia reportada para 2 hp.
3. O dado correto existe no arquivo (`X099_DE_time`); é só a seleção de chave
   que erra.

### Achado 3 — Quatro arquivos de falha sem assinatura observável

`197.mat` a `200.mat` (pista externa, 0,014", as quatro cargas) **não
apresentam pico em BPFO nem no sinal original a 12 kHz**. A razão pico/mediana
do espectro de envelope na banda de ressonância fica entre 3,0 e 6,8 — dentro do
ruído. Para comparação, os demais arquivos de pista externa ficam entre 1.097 e
20.980.

O RMS confirma: 0,094–0,101 contra 0,56–0,59 dos outros ensaios de pista
externa. É um comportamento conhecido dessa faixa do CWRU e não um erro de
download — os arquivos são íntegros, o defeito é que não se manifestou no sinal.

**Consequência:** 4 dos 22 arquivos de falha (18%) chegam ao modelo com rótulo
de falha sobre um sinal em que a falha não é observável. São rótulos corretos na
bancada e incorretos do ponto de vista do que o sensor mede. O modelo é obrigado
a decorá-los.

### Achado 4 — Desbalanceamento estrutural de 1 : 7,4

A classe normal representa 11,9% do sinal. O `build.py` compensa com
`class_weight`, mas peso não cria informação: são **59 janelas independentes**
de rolamento saudável contra 436 de falha.

> Nota: o comentário original do `build.py` atribuía o desbalanceamento a
> durações diferentes ("5 s no normal, 10 s nas falhas"). Isso vale só para
> `97.mat`, que de fato tem 5,08 s; os outros 27 ensaios têm ≈ 10 s. O
> desbalanceamento vem da **contagem de arquivos** (4 contra 24), porque o CWRU
> só publica 4 ensaios de baseline. O comentário foi corrigido.

### Achado 5 — Os relatórios estão dessincronizados dos dados

`relatorios/relatorio_metricas.md` e `README.md` citam 28 arquivos, 279,4 s e
545 janelas independentes. Os dados em disco hoje dão **25 arquivos, 254,0 s e
496 janelas**. O relatório foi gerado quando os três arquivos de 0 hp estavam
presentes. Qualquer número citado no TCC precisa vir de uma execução com o
inventário fechado.

---

## 4. O que a decimação custa

Este é o achado central, e vale mais do que os quatro anteriores somados.

O MPU6050 atualiza os registradores a no máximo 1 kHz. O `build.py` decima os
sinais de 12/48 kHz para 1 kHz para que o modelo veja em bancada o mesmo
conteúdo espectral que verá em campo. O filtro anti-aliasing remove tudo acima
de 500 Hz — e é lá que mora quase todo o sinal:

| Classe | 0–500 Hz | 500 Hz–2 kHz | 2–5 kHz | > 5 kHz | **Descartado** |
|---|---|---|---|---|---|
| `0_normal` | 22,4% | 18,8% | 39,3% | 19,6% | **77,6%** |
| `1_inner_race` | 1,3% | 9,4% | 89,2% | 0,1% | **98,7%** |
| `2_outer_race` | 1,8% | 7,5% | 88,8% | 2,0% | **98,2%** |

A concentração em 2–5 kHz não é coincidência: é a **ressonância estrutural do
mancal**, excitada pelos impactos do defeito. O diagnóstico clássico de
rolamentos funciona demodulando exatamente essa banda. Ao decimar para 1 kHz,
ela desaparece inteira.

O efeito é visível nos indicadores clássicos de impacto:

| Classe | Curtose a 12/48 kHz | Curtose a 1 kHz |
|---|---|---|
| `0_normal` | −0,06 | −0,61 |
| `1_inner_race` | **+8,98** | −0,26 |
| `2_outer_race` | **+8,45** | +0,58 |

Curtose alta é a assinatura de um sinal impulsivo. Depois da decimação, **as três
classes ficam gaussianas e indistinguíveis por esse critério**. O mesmo vale para
amplitude: a faixa de RMS da classe normal a 1 kHz (0,0294–0,0350) fica *dentro*
da faixa das duas classes de falha (0,0194–0,0765), e vários ensaios de falha têm
RMS **menor** que o rolamento saudável.

> Este é o motivo pelo qual o modelo colapsa em 0 hp. Sem carga radial, o impacto
> no defeito é mais fraco; o pouco que restaria na banda de 0–500 Hz cai abaixo
> do ruído, e a janela passa a parecer um rolamento normal. Não é um defeito do
> treino — é o limite físico do sensor escolhido.

---

## 5. Então o dataset ainda serve?

Serve. A prova está no espectro de **envelope**, medido na frequência
característica de cada defeito (razão pico/mediana; ruído puro fica abaixo de 10):

| Classe / severidade | Banda 2–5 kHz | Após decimar a 1 kHz |
|---|---|---|
| Interna, 0,007" | 3.671 – 5.633 | 206 – 1.318 |
| Interna, 0,014" | 513 – 1.264 | 84 – 2.607 |
| Interna, 0,021" | 1.138 – 2.861 | 1.762 – 6.489 |
| Externa, 0,007" | 18.333 – 20.980 | 74 – 346 |
| Externa, 0,014" | **3 – 7** | 40 – 100 |
| Externa, 0,021" | 1.097 – 1.600 | 1.036 – 1.302 |

Dezoito dos 22 arquivos de falha mantêm a assinatura bem acima do piso de ruído
depois da decimação — porque BPFI (156–162 Hz) e BPFO (103–107 Hz) estão dentro
da banda passante. O que se perde é **margem**: no caso mais severo
(`131.mat`), a razão cai de 20.980 para 73,8 — um fator de 284.

A conclusão correta é:

> A 1 kHz o problema continua **solúvel, mas com margem estreita**. A informação
> que resta é a periodicidade dos impactos na banda de 0–500 Hz, não a energia
> deles. Um classificador que aprenda amplitude vai falhar; um que aprenda
> periodicidade sobrevive. Isso é um argumento a favor da CNN 1D — e contra
> qualquer limiar simples de RMS.

---

## 6. Tratativas possíveis

Ordenadas por relação entre ganho esperado e custo. Nenhuma foi aplicada — esta
seção é o plano, não o registro.

### Correções (custo baixo, sem discussão) — **aplicadas**

| # | Ação | Efeito | Situação |
|---|---|---|---|
| T1 | Selecionar a chave `_DE_time` pelo **número do arquivo**, não pela ordem do dicionário | Elimina o Achado 2. Recupera o baseline de 2 hp e fecha o vazamento na validação por carga | ✅ |
| T2 | **Abortar** se o inventário não fechar com o catálogo | Impede treinar em silêncio com dataset incompleto (Achado 1) | ✅ na etapa 3 |
| T3 | Rebaixar `97/105/130.mat` | Fecha as três células vazias de 0 hp | ✅ 28/28 |
| T4 | Regerar `relatorio_metricas.md` e sincronizar o `README.md` | Corrige o Achado 5 | ⏳ requer novo `build.py` |
| T5 | Corrigir o comentário do `build.py` sobre as durações | Achado 4 | ✅ |

### Melhorias de dataset (custo médio, exigem decisão)

| # | Ação | Efeito esperado | Risco |
|---|---|---|---|
| T6 | Adicionar a classe **esfera** (`ball fault`, arquivos 118–121, 185–188, 222–225) | Quarta classe, +12 arquivos, diagnóstico mais completo | Muda o `num_classes`, exige retreino e nova validação embarcada |
| T7 | Incluir o canal **FE** (`fan end`) como segundo exemplo da mesma condição | Dobra o sinal disponível sem novos downloads | Sensor em posição diferente; é aumento de dados, não de diversidade real |
| T8 | Trazer os ensaios de **48 kHz Drive End** (arquivos 109–112 etc.) | Mais gravações independentes das mesmas condições | Decimação por 48 em vez de 12; o `resample_to` já suporta |
| T9 | **Excluir ou marcar** `197–200.mat` | Remove 18% de rótulos não observáveis (Achado 3) | Reduz o dataset; precisa ser declarado no TCC como exclusão justificada — ✅ **aplicada**, por critério medido e não por lista negra |
| T10 | Ampliar a classe normal com os ensaios **Normal Baseline a 48 kHz** que faltam | Ataca o desbalanceamento de 1 : 7,4 | O CWRU só tem 4 ensaios de baseline; o teto é estrutural |

### Mudanças de representação (custo alto, maior potencial)

| # | Ação | Efeito esperado |
|---|---|---|
| T11 | **Demodular antes de decimar**: filtrar 2–5 kHz, extrair o envelope de Hilbert e *só então* decimar para 1 kHz | Preserva a periodicidade dos impactos em vez de descartá-la. É o que a Seção 4 aponta como a perda dominante. **Não implementável no ESP32 com o MPU6050** — o sensor não entrega 12 kHz — mas define o limite superior teórico e é a justificativa quantitativa para a escolha de sensor |
| T12 | Trocar o MPU6050 por um acelerômetro de banda larga (ADXL1002, 11 kHz; IIS3DWB, 6 kHz) | Elimina a causa raiz. Torna T11 viável em campo e deve resolver a falha em 0 hp |
| T13 | Alimentar o modelo com o **espectro** da janela em vez da série temporal | FFT de 512 pontos no ESP32 custa ~1 ms; pode facilitar o aprendizado da periodicidade |
| T14 | Aumento de dados: ruído gaussiano, deslocamento circular, escala de amplitude | Ataca a fragilidade entre cargas sem novos dados. Barato de testar |
| T15 | Validação **leave-one-severity-out** além de leave-one-load-out | Mede se o modelo generaliza para um diâmetro de defeito inédito — mais próximo do uso real, em que a severidade é desconhecida |

### Recomendação

Executar **T1–T5** antes de qualquer coisa: são correções, não escolhas, e
mudam os números que já estão no TCC. Depois, **T14 e T15** — são baratos e
respondem à pergunta que mais pesa na banca ("isso generaliza?"). **T11** vale
como capítulo de análise mesmo sem ser embarcável: é ele que fecha o argumento
de que a limitação medida é do sensor, não do método.

---

## 7. Limitações a declarar

Independentemente das tratativas acima, permanecem verdadeiras:

1. **Defeitos usinados.** Eletroerosão produz um defeito localizado e de bordas
   limpas. Falhas reais evoluem por fadiga, de forma progressiva e distribuída.
2. **Uma única bancada.** Todos os ensaios vêm da mesma máquina, mesmo
   alinhamento, mesma montagem. Não há variação entre instalações.
3. **Faixa de rotação estreita.** 1.721–1.797 rpm — uma variação de 4,4%. O
   modelo nunca viu partida, parada ou rotação variável.
4. **Sem condições de falha compostas.** Um rolamento real pode ter defeito em
   mais de um elemento, ou defeito somado a desbalanceamento e desalinhamento.

---

## Ver também

- [`../dataset_engineering/amostras/LEIA-ME.md`](../dataset_engineering/amostras/LEIA-ME.md)
  — o dataset em CSV, antes e depois das tratativas: catálogo dos 28 ensaios,
  o sinal na taxa original, a janela que entra na rede e a correção T1 lado a lado
- [`../dataset_engineering/README.md`](../dataset_engineering/README.md) — o fluxo
  que aplica as tratativas e mede o efeito de cada uma
- [`pipeline.md`](pipeline.md) — como esses dados viram firmware
- [`protocolo_validacao.md`](protocolo_validacao.md) — o que é validado no hardware
- `tools/analise_dataset.py` — script que gera todos os números acima
