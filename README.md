# Manutenção Preditiva de Rolamentos com TinyML no ESP32

TCC — Impacta. Classificação de falhas em rolamentos a partir de sinais de
vibração, com uma CNN 1D quantizada em int8 rodando em um ESP32 com acelerômetro
MPU6050.

O pipeline vai do dataset bruto ao firmware: carrega os arquivos do
[CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter),
decima de 12/48 kHz para 1 kHz (teto do MPU6050), treina, quantiza para int8,
gera os arquivos C e grava no microcontrolador — junto com um conjunto de
validação que roda **dentro** do ESP32.

---

## Resultados

| Métrica | Valor |
|---|---|
| Acurácia — divisão temporal intra-arquivo | 99,77% |
| Acurácia — deixando uma carga de fora | 86,01% |
| ↳ apenas cargas 1–3 hp | 99,50% |
| Inferência no ESP32 | 9,4 ms |
| Tensor arena | 4,7 KB de 24 KB |
| Modelo em flash | 15 KB |
| Ciclo útil (janela de 512 ms) | 1,8% |

> **Leia os dois primeiros números juntos.** A divisão intra-arquivo coloca
> janelas da mesma gravação em treino e teste; ela mede consistência dentro da
> condição de operação, não generalização. O número de 86,01% é o realista — e
> ele esconde uma falha específica: **com o rolamento descarregado (0 hp) a
> detecção cai para 43%**, com falhas sendo classificadas como normal. Detalhes
> em [`docs/protocolo_validacao.md`](docs/protocolo_validacao.md) §6.

---

## Requisitos

- **Python 3.11+** (desenvolvido em 3.13)
- **PlatformIO CLI** para a parte embarcada — `pip install platformio`
- **ESP32 DevKit** + **MPU6050** (opcional: o pipeline Python roda sozinho)

Ligação do sensor:

| MPU6050 | ESP32 |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SDA | GPIO 21 |
| SCL | GPIO 22 |

---

## Reproduzindo o treino

```bash
# 1. Ambiente
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt

# 2. Dataset (~102 MB, 28 arquivos .mat)
python download_cwru.py

# 3. Treino, quantização e geração dos artefatos C
python build.py
```

Os arquivos `.mat` **não são versionados** — são dados públicos e o
`download_cwru.py` baixa exatamente os 28 usados no treino: 4 níveis de carga
(0–3 hp) e 3 diâmetros de defeito (0,007" / 0,014" / 0,021") por classe de falha.
O download é idempotente; rodar de novo só busca o que falta.

O `build.py` executa seis etapas e é **determinístico** (`SEED = 42` aplicado a
NumPy, `random` e TensorFlow): a mesma entrada gera artefatos byte a byte
idênticos. Ele produz:

| Saída | Descrição |
|---|---|
| `arduino_deploy/tinyml_esp32/src/model_tflite.cpp` | modelo quantizado como array C |
| `arduino_deploy/tinyml_esp32/include/model_params.h` | janela, classes, normalização, taxa de amostragem |
| `arduino_deploy/tinyml_esp32/src/test_vectors.cpp` | conjunto de validação embarcado |
| `relatorios/relatorio_metricas.md` | métricas, gráficos e limites de hardware |

---

## Gravando o ESP32

```bash
cd arduino_deploy/tinyml_esp32
pio run -t upload
pio device monitor
```

O firmware roda um diagnóstico completo no boot (varredura I2C, `WHO_AM_I`,
custo de leitura, taxa de saída real do sensor via `DATA_RDY`) e depois classifica
uma janela a cada 2 s, imprimindo estatísticas de amostragem, quantização e
inferência.

### Comandos do console serial

| Tecla | Ação |
|---|---|
| `h` | ajuda |
| `s` | reexecuta o diagnóstico do sensor |
| `d` | modo verboso (amostras cruas) |
| `x` `y` `z` `m` | eixo enviado ao modelo (`m` = módulo) |
| `o` | liga/desliga a remoção do nível DC (gravidade) |
| `f` | alterna fs: 250 → 500 → 1000 → 2000 Hz |
| `r` | despeja a janela atual em CSV, para FFT |
| `t` | **valida o deploy com o conjunto de teste do CWRU** |
| `p` | pausa/retoma a inferência |

O comando `t` reprocessa dentro do ESP32 as mesmas janelas de teste que geraram a
acurácia do relatório e compara com a saída do interpretador TFLite do PC —
inclusive verificando, por hash, que a entrada int8 montada no microcontrolador é
byte a byte idêntica à do PC. É o que separa "o modelo funciona no notebook" de
"o sistema embarcado funciona".

---

## Validações adicionais

```bash
# Generalização para uma condição de carga inédita (leave-one-load-out)
python tools/validacao_por_carga.py

# Sensibilidade da softmax int8 a 1 LSB nos logitos
python tools/sensibilidade_softmax.py

# Testes do pipeline
pytest tests/ -q
```

---

## Estrutura

```
build.py                      pipeline completo (6 etapas)
download_cwru.py              catálogo e download do dataset
src/pipeline/
  data_processor.py           carga, decimação, janelamento, split sem vazamento
  model_builder.py            arquitetura da CNN 1D
  quantizer.py                conversão para int8 + utilitários de quantização
  c_generator.py              geração dos artefatos C
  evaluator.py                métricas, gráficos e relatório
tools/                        validações independentes
tests/                        testes do pipeline
arduino_deploy/tinyml_esp32/  projeto PlatformIO (firmware)
docs/protocolo_validacao.md   protocolo de validação e resultados medidos
relatorios/                   relatório e gráficos gerados
data/                         dataset (não versionado)
```

---

## Documentação

- **[`docs/pipeline.md`](docs/pipeline.md)** — como a pipeline funciona: o que
  cada uma das seis etapas faz, por que faz assim, e como o resultado chega ao
  ESP32. Comece por aqui para entender o código.
- **[`docs/protocolo_validacao.md`](docs/protocolo_validacao.md)** — o que é
  validado, como reproduzir cada ensaio, resultados medidos no hardware e as
  limitações a declarar.
- **[`tinyml_relatorio.md`](tinyml_relatorio.md)** — justificativa das decisões de
  arquitetura da rede e da quantização, especificação dos tensores e requisitos
  de integração em C++.
- **`relatorios/relatorio_metricas.md`** — gerado pelo `build.py` a cada execução.

---

## Limitações conhecidas

1. **Rolamento descarregado (0 hp).** A detecção cai para 43%: mais da metade das
   janelas com falha é classificada como normal. Causa provável: sem carga
   radial, o impacto no defeito é fraco, e a decimação para 1 kHz descarta a
   banda de 2–5 kHz onde ele ainda seria visível.
2. **Banda de frequência.** O MPU6050 não passa de 1 kHz, contra os 12 kHz do
   dataset. É limitação do sensor, não do modelo.
3. **Faixa de amplitude.** O quantizador foi calibrado sobre o CWRU; sinais fora
   de ~0,1x a 2,0x a amplitude de treino são ceifados. O firmware alerta.
4. **Defeitos usinados.** O CWRU usa defeitos feitos por eletroerosão. Falhas
   reais evoluem de forma progressiva e podem ter assinatura distinta.

---

## Fonte dos dados

Case Western Reserve University Bearing Data Center.
https://engineering.case.edu/bearingdatacenter
