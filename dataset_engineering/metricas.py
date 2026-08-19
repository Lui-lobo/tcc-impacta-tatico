"""Medidas do dataset. Usadas identicamente no "antes" e no "depois".

Este modulo existe para garantir que a comparacao pre/pos curadoria seja
honesta. Se a etapa 2 e a etapa 4 calculassem separabilidade com codigo
proprio, qualquer diferenca entre elas poderia vir da medida e nao do dado.
Aqui ha uma implementacao unica; as duas etapas apenas a chamam sobre conjuntos
diferentes.

O QUE E MEDIDO
--------------
tempo      - RMS, curtose, fator de crista. Curtose alta = sinal impulsivo, que
             e a assinatura fisica de um defeito de rolamento.
espectro   - fracao de energia por banda, para quantificar o que a decimacao
             para 1 kHz descarta.
envelope   - razao pico/mediana do espectro de envelope na frequencia
             caracteristica do defeito (BPFI/BPFO). E a medida direta de
             "a assinatura ainda esta la".
sonda      - acuracia de um classificador LINEAR sobre atributos simples, com
             validacao deixando uma carga de fora. Nao e o modelo do projeto: e
             um termometro barato e deterministico da separabilidade do dado.
"""
import numpy as np
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataset_engineering import config

# Numero de sub-bandas em que a faixa util (0 a Nyquist) e dividida para compor
# os atributos da sonda.
N_SUBBANDAS = 8


# ---------------------------------------------------------------------------
# Dominio do tempo
# ---------------------------------------------------------------------------
def estatisticas_tempo(x):
    rms = float(np.sqrt(np.mean(x ** 2)))
    pico = float(np.max(np.abs(x)))
    return {
        "rms": rms,
        "pico": pico,
        "curtose": float(stats.kurtosis(x)),
        "crista": float(pico / rms) if rms > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Dominio da frequencia
# ---------------------------------------------------------------------------
def bandas_energia(x, fs, nyquist_alvo=None):
    """Fracao da energia total em cada banda, e o total descartado ao decimar."""
    nyquist_alvo = nyquist_alvo or config.NYQUIST_ALVO_HZ
    f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x), 8192))
    total = pxx.sum()
    if total <= 0:
        return {}

    limites = [(0, nyquist_alvo), (nyquist_alvo, 2000), (2000, 5000),
               (5000, np.inf)]
    nomes = [f"0-{nyquist_alvo}", f"{nyquist_alvo}-2k", "2k-5k", ">5k"]
    fracoes = {nome: float(pxx[(f >= a) & (f < b)].sum() / total)
               for nome, (a, b) in zip(nomes, limites)}
    # Tudo acima do Nyquist de destino e removido pelo filtro anti-aliasing.
    fracoes["descartado"] = float(pxx[f >= nyquist_alvo].sum() / total)
    return fracoes


def razao_envelope(x, fs, alvo, banda=None, tolerancia_hz=3.0):
    """Razao pico/mediana do espectro de envelope em torno de `alvo`.

    Um defeito de rolamento nao produz um tom em `alvo`: produz impulsos
    periodicos que MODULAM uma ressonancia estrutural de alta frequencia. A FFT
    direta do sinal nao mostra isso; a demodulacao (envelope de Hilbert) mostra.

    Com `banda`, o sinal e primeiro filtrado na faixa de ressonancia - que e o
    procedimento classico. Sem `banda`, a medida e feita sobre o sinal inteiro,
    que e o unico caminho possivel depois da decimacao para 1 kHz (a banda de
    ressonancia ja nao existe).
    """
    if alvo <= 0 or len(x) < 64:
        return float("nan")

    if banda is not None:
        alta = min(0.99, banda[1] / (fs / 2))
        baixa = banda[0] / (fs / 2)
        if not 0 < baixa < alta:
            return float("nan")
        b, a = signal.butter(4, [baixa, alta], btype="band")
        x = signal.filtfilt(b, a, x)

    envelope = np.abs(signal.hilbert(x))
    envelope = envelope - envelope.mean()
    f, pxx = signal.welch(envelope, fs=fs, nperseg=min(len(envelope), 16384))

    util = (f > 5) & (f < config.NYQUIST_ALVO_HZ)
    if not util.any():
        return float("nan")
    f, pxx = f[util], pxx[util]

    vizinhanca = np.abs(f - alvo) < tolerancia_hz
    if not vizinhanca.any():
        return float("nan")
    return float(pxx[vizinhanca].max() / np.median(pxx))


# ---------------------------------------------------------------------------
# Medida por ensaio
# ---------------------------------------------------------------------------
def medir_ensaio(registro):
    """Todas as medidas de um ensaio, no sinal original e no decimado."""
    ensaio = registro["ensaio"]
    bruto, dec = registro["bruto"], registro["dec"]
    alvo = ensaio.frequencia_defeito

    medida = {
        "numero": ensaio.numero,
        "classe": ensaio.classe,
        "pasta": ensaio.pasta,
        "severidade": ensaio.severidade,
        "carga": ensaio.carga,
        "rpm_nominal": ensaio.rpm,
        "rpm_medido": registro.get("rpm_medido"),
        "fs_origem": ensaio.taxa_origem,
        "amostras_origem": int(len(bruto)),
        "amostras_dec": int(len(dec)),
        "duracao_s": float(len(dec) / config.TAXA_ALVO_HZ),
        "frequencia_defeito": float(alvo),
        "chave_usada": registro.get("chave_usada"),
        "chave_ingenua": registro.get("chave_ingenua"),
        "divergente": bool(registro.get("divergente", False)),
        "origem": {**estatisticas_tempo(bruto),
                   **bandas_energia(bruto, ensaio.taxa_origem)},
        "decimado": estatisticas_tempo(dec),
    }

    if ensaio.classe != 0:
        medida["envelope_ressonancia"] = razao_envelope(
            bruto, ensaio.taxa_origem, alvo, banda=config.BANDA_RESSONANCIA_HZ)
        medida["envelope_decimado"] = razao_envelope(
            dec, config.TAXA_ALVO_HZ, alvo)
        medida["observavel"] = (
            medida["envelope_ressonancia"] >= config.PISO_ENVELOPE)
    else:
        medida["envelope_ressonancia"] = float("nan")
        medida["envelope_decimado"] = float("nan")
        medida["observavel"] = True

    return medida


def medir_conjunto(registros):
    return [medir_ensaio(r) for r in registros]


# ---------------------------------------------------------------------------
# Escala e balanco
# ---------------------------------------------------------------------------
def escala(medidas):
    """Quanto sinal independente existe, no total e por classe."""
    amostras = sum(m["amostras_dec"] for m in medidas)
    resumo = {
        "arquivos": len(medidas),
        "amostras": amostras,
        "duracao_s": amostras / config.TAXA_ALVO_HZ,
        "janelas_independentes": amostras // config.TAMANHO_JANELA,
        "por_classe": {},
    }
    for classe, pasta in config.CLASSES.items():
        grupo = [m for m in medidas if m["classe"] == classe]
        n = sum(m["amostras_dec"] for m in grupo)
        resumo["por_classe"][pasta] = {
            "arquivos": len(grupo),
            "duracao_s": n / config.TAXA_ALVO_HZ,
            "fracao": (n / amostras) if amostras else 0.0,
            "janelas_independentes": n // config.TAMANHO_JANELA,
        }

    fracoes = [v["fracao"] for v in resumo["por_classe"].values() if v["fracao"] > 0]
    resumo["desbalanceamento"] = (max(fracoes) / min(fracoes)) if fracoes else 0.0
    return resumo


def sobreposicao_rms(medidas):
    """As faixas de RMS pos-decimacao das classes se sobrepoem?

    Se a faixa da classe normal invade a das falhas, nenhum limiar de amplitude
    separa as duas - e o classificador precisa aprender forma, nao energia.
    """
    faixas = {}
    for classe, pasta in config.CLASSES.items():
        valores = [m["decimado"]["rms"] for m in medidas if m["classe"] == classe]
        if valores:
            faixas[pasta] = (min(valores), max(valores))

    normal = faixas.get(config.CLASSES[0])
    invadidas = []
    if normal:
        for pasta, (a, b) in faixas.items():
            if pasta != config.CLASSES[0] and a <= normal[1] and b >= normal[0]:
                invadidas.append(pasta)
    return {"faixas": faixas, "sobrepostas": invadidas}


# ---------------------------------------------------------------------------
# Sonda de separabilidade
# ---------------------------------------------------------------------------
def atributos(janelas, fs=None):
    """Atributos simples por janela: estatisticas de tempo + energia por banda.

    Deliberadamente rasos. A pergunta que a sonda responde nao e "qual a melhor
    acuracia possivel", e sim "a informacao esta acessivel no dado?". Atributos
    sofisticados confundiriam as duas coisas.
    """
    fs = fs or config.TAXA_ALVO_HZ
    x = janelas - janelas.mean(axis=1, keepdims=True)

    rms = np.sqrt(np.mean(x ** 2, axis=1))
    seguro = np.maximum(rms, 1e-12)
    pico = np.max(np.abs(x), axis=1)
    tempo = np.column_stack([
        np.log10(seguro),
        stats.kurtosis(x, axis=1),
        stats.skew(x, axis=1),
        pico / seguro,
        np.mean(np.abs(np.diff(np.signbit(x), axis=1)), axis=1),  # taxa de cruzamento por zero
    ])

    espectro = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2
    bandas = np.array_split(espectro[:, 1:], N_SUBBANDAS, axis=1)
    energia = np.column_stack([np.log10(b.sum(axis=1) + 1e-12) for b in bandas])
    # Energia relativa: remove a escala global e deixa so a FORMA do espectro,
    # que e o que distingue as classes depois que a decimacao achata as
    # amplitudes.
    energia = energia - energia.mean(axis=1, keepdims=True)

    return np.column_stack([tempo, energia])


def sonda_linear(registros, verbose=False):
    """Separabilidade por regressao logistica, deixando uma carga de fora.

    Protocolo identico ao do tools/validacao_por_carga.py, para que os numeros
    conversem: treina em tres niveis de carga, testa no quarto, repete quatro
    vezes. Janelas independentes dos dois lados.

    A sonda nao substitui a CNN. Ela responde em segundos, e deterministicamente,
    se a curadoria tornou o dado mais separavel - antes de gastar um build.py
    inteiro para descobrir.
    """
    from dataset_engineering.leitura import janelas_independentes

    por_carga = {c: {"X": [], "y": []} for c in config.CARGAS}
    for r in registros:
        ensaio = r["ensaio"]
        jan = janelas_independentes(r["dec"])
        if len(jan) == 0:
            continue
        por_carga[ensaio.carga]["X"].append(atributos(jan))
        por_carga[ensaio.carga]["y"].append(
            np.full(len(jan), ensaio.classe, dtype=int))

    for c in config.CARGAS:
        d = por_carga[c]
        d["X"] = np.vstack(d["X"]) if d["X"] else np.empty((0, 0))
        d["y"] = np.concatenate(d["y"]) if d["y"] else np.empty(0, dtype=int)

    rodadas = []
    for teste in config.CARGAS:
        X_te, y_te = por_carga[teste]["X"], por_carga[teste]["y"]
        treino = [c for c in config.CARGAS if c != teste and len(por_carga[c]["y"])]
        if len(y_te) == 0 or not treino:
            continue
        X_tr = np.vstack([por_carga[c]["X"] for c in treino])
        y_tr = np.concatenate([por_carga[c]["y"] for c in treino])
        if len(np.unique(y_tr)) < 2:
            continue

        escalador = StandardScaler().fit(X_tr)
        modelo = LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=config.SEMENTE)
        modelo.fit(escalador.transform(X_tr), y_tr)
        preditos = modelo.predict(escalador.transform(X_te))

        falhas, normais = y_te > 0, y_te == 0
        rodadas.append({
            "carga": teste,
            "n_teste": int(len(y_te)),
            "n_treino": int(len(y_tr)),
            "acuracia": float(np.mean(preditos == y_te)),
            # Deteccao e falso alarme separam as duas tarefas embutidas no
            # problema: confundir pista interna com externa e erro de
            # DIAGNOSTICO; chamar um rolamento defeituoso de normal e erro de
            # DETECCAO, com consequencia operacional completamente diferente.
            "deteccao": (float(np.mean(preditos[falhas] > 0))
                         if falhas.any() else float("nan")),
            "falso_alarme": (float(np.mean(preditos[normais] > 0))
                             if normais.any() else float("nan")),
            "classes_no_teste": sorted(int(c) for c in np.unique(y_te)),
        })
        if verbose:
            r = rodadas[-1]
            print(f"   carga {teste} hp: acuracia={r['acuracia']*100:.2f}% "
                  f"deteccao={r['deteccao']*100:.2f}%")

    if not rodadas:
        return {"rodadas": [], "acuracia_media": float("nan"),
                "deteccao_media": float("nan")}

    return {
        "rodadas": rodadas,
        "acuracia_media": float(np.mean([r["acuracia"] for r in rodadas])),
        "acuracia_desvio": float(np.std([r["acuracia"] for r in rodadas])),
        "deteccao_media": float(np.nanmean([r["deteccao"] for r in rodadas])),
        "falso_alarme_medio": float(np.nanmean(
            [r["falso_alarme"] for r in rodadas])),
    }


def diagnostico_completo(registros, verbose=False):
    """Pacote unico de medidas. E o que a etapa 2 e a etapa 4 comparam."""
    medidas = medir_conjunto(registros)
    return {
        "medidas": medidas,
        "escala": escala(medidas),
        "rms": sobreposicao_rms(medidas),
        "sonda": sonda_linear(registros, verbose=verbose),
        "divergentes": [m["numero"] for m in medidas if m["divergente"]],
        "nao_observaveis": [m["numero"] for m in medidas
                            if m["classe"] != 0 and not m["observavel"]],
    }
