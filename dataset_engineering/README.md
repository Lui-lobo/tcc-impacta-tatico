# Fluxo de Engenharia de Dados

Do `.mat` bruto do CWRU ao dataset curado que o `build.py` consome.

```bash
python -m dataset_engineering.executar
```

Quatro etapas encadeadas. Cada uma roda sozinha, e todas escrevem em
`relatorios/`.

| Etapa | Comando | O que faz |
|---|---|---|
| 1 · Leitura | `python -m dataset_engineering.etapa1_leitura` | Lê e **demonstra** o dataset: inventário, canais, cobertura do plano fatorial e quatro figuras |
| 2 · Diagnóstico | `python -m dataset_engineering.etapa2_diagnostico` | Mede o **ANTES**, lendo como o `build.py` lê hoje. Grava `diagnostico_antes.json` |
| 3 · Curadoria | `python -m dataset_engineering.etapa3_curadoria` | Aplica as tratativas e materializa `data_curado/` |
| 4 · Comparação | `python -m dataset_engineering.etapa4_comparacao` | Mede o **DEPOIS** e faz a ablação por tratativa |
| + · Amostras | `python -m dataset_engineering.exportar_amostras` | Exporta o dataset em CSV, antes e depois — [`amostras/`](amostras/LEIA-ME.md) |

Saída consolidada: [`relatorios/engenharia_dados.md`](relatorios/engenharia_dados.md).

---

## Por que existe

O projeto tratava o dataset como entrada dada. A auditoria em
[`../docs/dataset.md`](../docs/dataset.md) encontrou quatro problemas reais —
canal errado em um arquivo, conteúdo duplicado, ensaios sem assinatura de
defeito e relatórios dessincronizados. Este fluxo os corrige de forma
reproduzível, e mede o efeito de cada correção em vez de afirmá-lo.

---

## Tratativas aplicadas

| # | Tratativa | Por quê |
|---|---|---|
| **T1** | Selecionar o canal pelo **número do arquivo** | `99.mat` carrega também as variáveis de `98.mat`. Pegar "a primeira chave `_DE_time`" entregava a série de 1 hp rotulada como 2 hp |
| **T2** | Inventário obrigatório | A curadoria para se faltar arquivo. Treinar em silêncio com dataset incompleto foi o que produziu os números dessincronizados |
| **T3** | Deduplicação por hash | Efeito colateral de T1; a verificação fica porque um download novo pode reintroduzir o problema |
| **T9** | Quarentena por **observabilidade medida** | Ensaios de falha cuja frequência característica não se destaca do ruído — nem no sinal original — carregam rótulo que o sinal não sustenta |

T9 é um **critério**, não uma lista negra: mede a razão pico/mediana do espectro
de envelope em BPFI/BPFO, no sinal original demodulado em 2–5 kHz, e reprova o
que ficar abaixo do piso de ruído (`config.PISO_ENVELOPE`).

Para desligar a quarentena e treinar com tudo:

```bash
python -m dataset_engineering.executar --manter-suspeitos
```

---

## O que sai

```
data_curado/
  manifesto.json          metadados dos 28 ensaios, incluídos e em quarentena
  0_normal/097.npy        série já decimada para 1 kHz, float32
  1_inner_race/105.npy
  ...

dataset_engineering/relatorios/
  engenharia_dados.md     relatório consolidado (gerado, não escrito à mão)
  diagnostico_antes.json  medidas do dataset bruto
  comparacao.json         ablação antes/depois
  figuras/*.png           quatro figuras em 200 DPI

dataset_engineering/amostras/
  01_catalogo_bruto.csv   os 28 ensaios como vêm do CWRU
  02_catalogo_curado.csv  os mesmos 28, com status e motivo da curadoria
  03_sinal_bruto.csv      100 ms de três classes, na taxa original
  04_janela_modelo.csv    a janela de 512 amostras que entra na rede
  05_correcao_t1.csv      as duas séries que 99.mat pode entregar
```

Os CSVs de `amostras/` são **versionados** no repositório — são pequenos e
descrevem o dado. O `.mat` do CWRU não abre em planilha nem em editor de texto;
esses arquivos existem para que o dataset possa ser inspecionado, e não apenas
descrito. Ver [`amostras/LEIA-ME.md`](amostras/LEIA-ME.md).

As séries vão **já decimadas**. A decimação é determinística e cara; guardá-la
uma vez faz o `build.py`, o `tools/validacao_por_carga.py` e este fluxo
partirem do mesmo sinal, byte a byte.

O `manifesto.json` carrega classe, severidade, carga e rpm de cada ensaio —
metadados que antes estavam hard-coded em três arquivos diferentes, e que
habilitam validação estratificada sem duplicar mapas.

---

## Como o resto do projeto consome

`build.py` e `tools/validacao_por_carga.py` preferem `data_curado/` quando ele
existe, e caem de volta nos `.mat` brutos quando não existe. Nenhuma flag: a
detecção é automática, via `src/pipeline/dataset_curado.disponivel()`. O
`build.py` imprime na primeira linha qual fonte está em uso.

Para voltar ao comportamento antigo, apague `data_curado/`.

---

## Arquitetura

```
config.py       catálogo mestre dos 28 ensaios + parâmetros da curadoria
leitura.py      leitura corrigida (T1) + decimação, compartilhada por todas as etapas
metricas.py     todas as medidas — usadas IDENTICAMENTE no antes e no depois
etapa1..4.py    as quatro etapas
executar.py     orquestrador + gerador do relatório consolidado
```

`metricas.py` é único de propósito. Se a etapa 2 e a etapa 4 medissem
separabilidade com código próprio, qualquer diferença entre elas poderia vir da
medida em vez do dado.

### A sonda de separabilidade

`metricas.sonda_linear()` treina uma regressão logística sobre atributos rasos
(RMS, curtose, assimetria, fator de crista, taxa de cruzamento por zero e
energia em 8 sub-bandas), validando **deixando uma carga de fora** — o mesmo
protocolo do `tools/validacao_por_carga.py`.

Ela **não é o modelo do projeto**. É um termômetro barato e determinístico: roda
em segundos e diz se a curadoria tornou o dado mais separável, antes de gastar
um `build.py` inteiro para descobrir. Use-a para comparar conjuntos, nunca para
prever a acurácia da CNN.
