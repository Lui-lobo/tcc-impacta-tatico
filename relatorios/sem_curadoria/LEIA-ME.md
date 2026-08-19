# Artefatos do build SEM curadoria

Arquivados antes de reexecutar o `build.py` sobre o dataset curado, porque o
pipeline sobrescreve `relatorios/` a cada execução. São a **evidência da coluna
"sem curadoria"** de [`../../docs/comparativo_curadoria.md`](../../docs/comparativo_curadoria.md).

Não editar e não regerar.

| Arquivo | Origem |
|---|---|
| `relatorio_metricas.md` | `python build.py` sobre os 28 `.mat` brutos |
| `matriz_confusao.png` | idem |
| `historico_treinamento.png` | idem |
| `validacao_embarcada.txt` | comando `t` no console serial do ESP32 |

## Como identificar este build

| Item | Valor |
|---|---|
| Ensaios | 28 |
| Sinal a 1 kHz | 279,4 s |
| Janelas independentes | 545 |
| Janelas de teste | 1.313 (160 / 577 / 576) |
| Vetores na flash do ESP32 | 128 (**16 / 56 / 56**) |

A distribuição `16 / 56 / 56` é a assinatura deste deploy. O build com curadoria
produz `18 / 66 / 44`.
