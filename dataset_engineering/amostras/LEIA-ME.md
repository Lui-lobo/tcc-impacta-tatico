# Amostras do Dataset em CSV

O `.mat` do CWRU é um formato binário do MATLAB: não abre em editor de texto,
não abre em planilha, e nem sempre contém o que o nome do arquivo promete. Estes
CSVs existem para que o dataset possa ser **inspecionado**, e não apenas
descrito.

Gerados por:

```bash
python -m dataset_engineering.exportar_amostras
```

Rodam também no fim de `python -m dataset_engineering.executar`.

| Arquivo | Linhas | Responde |
|---|---|---|
| [`01_catalogo_bruto.csv`](01_catalogo_bruto.csv) | 28 | O que existe no dataset **como ele vem do CWRU**? |
| [`02_catalogo_curado.csv`](02_catalogo_curado.csv) | 28 | O que sobrou **depois da curadoria**, e por quê? |
| [`03_sinal_bruto.csv`](03_sinal_bruto.csv) | 7.200 | Como é o sinal na **taxa original** de gravação? |
| [`04_janela_modelo.csv`](04_janela_modelo.csv) | 512 | Como é a janela que **entra na rede**? |
| [`05_correcao_t1.csv`](05_correcao_t1.csv) | 512 | Que diferença fez **corrigir a leitura** de `99.mat`? |

**Formato:** separador `;`, decimal `,`. Abrem com duplo clique no Excel ou
LibreOffice em português. Em Python:

```python
pd.read_csv("01_catalogo_bruto.csv", sep=";", decimal=",")
```

---

## 01 e 02 — o antes e o depois, ensaio a ensaio

Os dois têm as mesmas 28 linhas, uma por ensaio do catálogo. A comparação entre
eles é o resumo da curadoria.

No **01**, a coluna `leitura_correta` marca `nao` em `99.mat`: o canal lido é
`X098_DE_time` quando deveria ser `X099_DE_time`. É o defeito que a tratativa
**T1** corrige — e note que `98.mat` e `99.mat` saem com RMS, curtose e fator de
crista **idênticos**, porque são literalmente a mesma série.

No **02**, a coluna `status` separa `incluido` de `quarentena`, e `motivo` diz
por quê. Os quatro ensaios em quarentena têm `envelope_ressonancia` entre 3,0 e
6,8 — dentro do ruído. Compare com os aprovados da mesma classe, que passam de
1.000.

A coluna `envelope_ressonancia` é a razão pico/mediana do espectro de envelope na
frequência característica do defeito, medida no sinal original demodulado em
2–5 kHz. É o critério de quarentena, e está exposto para ser conferido.

---

## 03 — o sinal como foi gravado

100 ms de três ensaios, todos a **3 hp** e severidade **0,021"**, para que a
comparação entre eles isole o tipo de defeito:

| Classe | Arquivo | Taxa |
|---|---|---|
| normal | `100.mat` | 48 kHz |
| pista interna | `212.mat` | 12 kHz |
| pista externa | `237.mat` | 12 kHz |

Formato **longo** (uma linha por amostra por classe) porque as classes têm taxas
diferentes e não compartilham base de tempo. Para plotar, filtre por
`classe_nome` e use `tempo_s` no eixo X.

100 ms cobrem cerca de 10 passagens de esfera em BPFO (~103 Hz) e 16 em BPFI
(~156 Hz) — o suficiente para a periodicidade dos impactos aparecer a olho.

---

## 04 — a janela que o modelo recebe

Os mesmos três ensaios, no mesmo instante da gravação, depois da decimação para
1 kHz: **512 amostras = 512 ms**, exatamente o tensor de entrada da CNN.

Formato **largo**: depois da decimação as três classes compartilham a base de
tempo, então dá para plotar as três numa planilha sem nenhum tratamento.

> **Compare 03 com 04 e o argumento central do projeto fica visível.** No 03 os
> impactos aparecem como picos nítidos e periódicos. No 04 eles sumiram — a
> decimação descarta ~98% da energia dos sinais de falha, incluindo a
> ressonância de 2–5 kHz que carrega a assinatura. O que resta é a
> periodicidade, não a energia. Ver [`../../docs/dataset.md`](../../docs/dataset.md) §4.

Os valores estão em **unidades físicas (g)**, antes da normalização. O
`build.py` ainda aplica `(x − média) / desvio` e quantiza para int8; essas duas
etapas estão descritas em [`../../docs/pipeline.md`](../../docs/pipeline.md) §1.5
e §3.

---

## 05 — a tratativa T1, lado a lado

As duas séries que `99.mat` pode entregar, na mesma base de tempo:

| Coluna | Série |
|---|---|
| `leitura_ingenua_X098_DE_time_g` | o que o pipeline lia — na verdade o ensaio de **1 hp** |
| `leitura_corrigida_X099_DE_time_g` | o ensaio de **2 hp**, que o arquivo promete |
| `diferenca_g` | a diferença amostra a amostra |

RMS de 0,034950 g contra 0,031286 g. **São sinais diferentes**, e a coluna
`diferenca_g` nunca é zero. É o que torna a T1 verificável por inspeção, em vez
de exigir confiança na descrição.

---

## Reproduzir

```bash
python download_cwru.py                          # 28 arquivos
python -m dataset_engineering.executar           # curadoria + estes CSVs
```

Os CSVs são **versionados** no repositório, ao contrário das séries completas.
São pequenos e descrevem o dado — é exatamente o que deve sobreviver a um
`git clone`.
