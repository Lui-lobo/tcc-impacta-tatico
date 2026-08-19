"""Auditoria do dataset CWRU: inventario, integridade e caracterizacao do sinal.

MOTIVACAO
---------
O restante do projeto trata o dataset como uma entrada dada: o build.py carrega
os .mat, decima e treina. Este script faz a pergunta anterior a essa - o que
exatamente ha nesses arquivos, e o que sobra deles depois da decimacao para
1 kHz imposta pelo MPU6050.

A resposta importa porque o teto de desempenho do sistema embarcado e definido
aqui, e nao na arquitetura da rede. Se a assinatura do defeito nao sobrevive a
decimacao, nenhuma CNN a recupera.

O QUE E VERIFICADO
------------------
1. Inventario     - quais arquivos do catalogo estao presentes em data/.
2. Integridade    - qual serie cada .mat de fato entrega ao pipeline, e se ha
                    conteudo duplicado entre arquivos.
3. Cobertura      - o plano fatorial classe x carga x severidade tem buracos?
4. Caracterizacao - RMS, curtose e fator de crista, no sinal original e depois
                    da decimacao.
5. Espectro       - distribuicao de energia por banda, para medir o que a
                    decimacao descarta.
6. Assinatura     - razao pico/mediana do espectro de ENVELOPE na frequencia
                    caracteristica do defeito (BPFI / BPFO), antes e depois da
                    decimacao. E a medida direta de "ainda ha o que aprender".

USO
---
    python tools/analise_dataset.py

Os resultados desta auditoria estao interpretados em docs/dataset.md.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.io
from scipy import signal, stats

import build as B
import download_cwru as D
from src.pipeline.data_processor import DataProcessor

# Geometria do rolamento SKF 6205-2RS JEM (lado do acionamento, Drive End).
# Multiplicadores da rotacao do eixo que dao a frequencia de passagem de esfera.
BPFI_MULT = 5.4152   # pista interna
BPFO_MULT = 3.5848   # pista externa

RPM_NOMINAL = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

CLASSES = ["0_normal", "1_inner_race", "2_outer_race"]

# arquivo -> carga (hp), extraido do catalogo do download_cwru.py.
CARGA_POR_ARQUIVO = {}
for _carga, _arquivos in enumerate([
    ["97", "105", "169", "209", "130", "197", "234"],
    ["98", "106", "170", "210", "131", "198", "235"],
    ["99", "107", "171", "211", "132", "199", "236"],
    ["100", "108", "172", "212", "133", "200", "237"],
]):
    for _n in _arquivos:
        CARGA_POR_ARQUIVO[f"{_n}.mat"] = _carga

# arquivo -> diametro do defeito usinado.
SEVERIDADE_POR_ARQUIVO = {}
for _sev, _arquivos in [
    ('0,007"', ["105", "106", "107", "108", "130", "131", "132", "133"]),
    ('0,014"', ["169", "170", "171", "172", "197", "198", "199", "200"]),
    ('0,021"', ["209", "210", "211", "212", "234", "235", "236", "237"]),
]:
    for _n in _arquivos:
        SEVERIDADE_POR_ARQUIVO[f"{_n}.mat"] = _sev

# Piso de ruido da razao pico/mediana do espectro de envelope. Abaixo disso a
# frequencia caracteristica nao se distingue do fundo.
PISO_ENVELOPE = 10.0


def titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def carregar_tudo(base_dir="data"):
    """Le cada .mat uma unica vez e devolve a serie original e a decimada."""
    proc = DataProcessor(window_size=B.WINDOW_SIZE, overlap=B.WINDOW_OVERLAP)
    registros = []
    for rotulo, pasta in enumerate(CLASSES):
        caminho = os.path.join(base_dir, pasta)
        if not os.path.isdir(caminho):
            continue
        for arquivo in sorted(glob.glob(os.path.join(caminho, "*.mat"))):
            nome = os.path.basename(arquivo)
            fs = B.SOURCE_RATE_BY_FILE.get(nome, B.DEFAULT_SOURCE_RATE_HZ)
            bruto = proc.load_mat_file(arquivo)
            registros.append({
                "rotulo": rotulo,
                "pasta": pasta,
                "nome": nome,
                "fs": fs,
                "carga": CARGA_POR_ARQUIVO.get(nome),
                "severidade": SEVERIDADE_POR_ARQUIVO.get(nome, "-"),
                "bruto": bruto,
                "dec": proc.resample_to(bruto, fs, B.TARGET_SAMPLE_RATE_HZ),
            })
    return registros


def inventario(base_dir="data"):
    """Compara o catalogo do download_cwru.py com o que existe em disco."""
    titulo("1. INVENTARIO - catalogo x arquivos presentes")

    ausentes = []
    total = 0
    for pasta, arquivos in D.CATALOGO.items():
        for nome, descricao in arquivos.items():
            total += 1
            if not os.path.exists(os.path.join(base_dir, pasta, nome)):
                ausentes.append((pasta, nome, descricao))

    print(f"Catalogados: {total} | Presentes: {total - len(ausentes)} | "
          f"Ausentes: {len(ausentes)}")
    if ausentes:
        print()
        print("AUSENTES (o build.py roda mesmo assim, com o dataset reduzido):")
        for pasta, nome, descricao in ausentes:
            print(f"   {pasta}/{nome:<10} {descricao}")
        print()
        print("   Reexecute 'python download_cwru.py' para tentar baixa-los.")
    return ausentes


def integridade(registros, base_dir="data"):
    """Verifica se cada .mat entrega a serie que o nome dele promete."""
    titulo("2. INTEGRIDADE - qual serie cada arquivo entrega ao pipeline")

    print(f"{'arquivo':<24}{'chave usada':<16}{'chave esperada':<16}"
          f"{'amostras':>10}  situacao")
    print("-" * 78)

    divergentes = []
    assinaturas = {}
    for reg in registros:
        caminho = os.path.join(base_dir, reg["pasta"], reg["nome"])
        mat = scipy.io.loadmat(caminho)
        usada = next((k for k in mat if "_DE_time" in k), "?")
        esperada = "X" + reg["nome"].replace(".mat", "").zfill(3) + "_DE_time"
        ok = usada == esperada
        if not ok:
            divergentes.append((reg["nome"], usada, esperada))
        # Assinatura do conteudo, para achar series identicas entre arquivos.
        chave = hash(reg["bruto"].tobytes())
        assinaturas.setdefault(chave, []).append(f"{reg['pasta']}/{reg['nome']}")
        print(f"{reg['pasta'] + '/' + reg['nome']:<24}{usada:<16}{esperada:<16}"
              f"{len(reg['bruto']):>10}  {'ok' if ok else '<<< DIVERGE'}")

    duplicados = [v for v in assinaturas.values() if len(v) > 1]
    print("-" * 78)
    if divergentes:
        print(f"{len(divergentes)} arquivo(s) entregam uma serie de OUTRO ensaio.")
        print("   load_mat_file() escolhe a primeira chave '_DE_time' que encontra;")
        print("   quando o .mat carrega mais de um ensaio, essa nem sempre e a certa.")
    if duplicados:
        print()
        print("CONTEUDO DUPLICADO (arquivos diferentes, mesma serie amostra a amostra):")
        for grupo in duplicados:
            print(f"   {' == '.join(grupo)}")
    if not divergentes and not duplicados:
        print("Nenhuma divergencia. Cada arquivo entrega a propria serie.")
    return divergentes, duplicados


def cobertura(registros):
    """Mostra os buracos do plano fatorial classe x severidade x carga."""
    titulo("3. COBERTURA - plano fatorial classe x severidade x carga")

    print(f"{'classe':<16}{'severidade':<12}" +
          "".join(f"{c} hp".rjust(10) for c in sorted(RPM_NOMINAL)))
    print("-" * 78)

    buracos = 0
    for rotulo, pasta in enumerate(CLASSES):
        severidades = sorted({r["severidade"] for r in registros
                              if r["rotulo"] == rotulo})
        for sev in severidades:
            celulas = []
            for carga in sorted(RPM_NOMINAL):
                presente = any(r["rotulo"] == rotulo and r["carga"] == carga
                               and r["severidade"] == sev for r in registros)
                buracos += 0 if presente else 1
                celulas.append(("sim" if presente else "AUSENTE").rjust(10))
            print(f"{pasta:<16}{sev:<12}" + "".join(celulas))

    print("-" * 78)
    print(f"Celulas vazias: {buracos}")
    if buracos:
        print("   Um buraco na carga X significa que a validacao leave-one-load-out")
        print("   testa aquela carga sem cobrir todas as classes/severidades.")
    return buracos


def caracterizacao(registros):
    """Estatisticas no dominio do tempo, antes e depois da decimacao."""
    titulo("4. CARACTERIZACAO - estatisticas antes x depois da decimacao")

    alvo = B.TARGET_SAMPLE_RATE_HZ
    print(f"{'arquivo':<24}{'carga':>7}{'dur_s':>7}"
          f"{'RMS_orig':>10}{'curt_orig':>10}"
          f"{'RMS_1k':>10}{'curt_1k':>9}{'crista_1k':>10}")
    print("-" * 78)

    for reg in registros:
        bruto, dec = reg["bruto"], reg["dec"]
        rms_o = np.sqrt(np.mean(bruto ** 2))
        rms_d = np.sqrt(np.mean(dec ** 2))
        print(f"{reg['pasta'] + '/' + reg['nome']:<24}{reg['carga']:>5} hp"
              f"{len(bruto) / reg['fs']:>7.2f}"
              f"{rms_o:>10.4f}{stats.kurtosis(bruto):>10.2f}"
              f"{rms_d:>10.4f}{stats.kurtosis(dec):>9.2f}"
              f"{np.max(np.abs(dec)) / rms_d:>10.2f}")

    print("-" * 78)
    print(f"{'classe':<16}{'n':>4}{'RMS_1k min':>12}{'RMS_1k med':>12}"
          f"{'RMS_1k max':>12}{'curt_orig':>11}{'curt_1k':>9}")
    faixas = {}
    for rotulo, pasta in enumerate(CLASSES):
        grupo = [r for r in registros if r["rotulo"] == rotulo]
        if not grupo:
            continue
        rms = [float(np.sqrt(np.mean(r["dec"] ** 2))) for r in grupo]
        faixas[pasta] = (min(rms), max(rms))
        print(f"{pasta:<16}{len(grupo):>4}{min(rms):>12.4f}"
              f"{float(np.median(rms)):>12.4f}{max(rms):>12.4f}"
              f"{np.mean([stats.kurtosis(r['bruto']) for r in grupo]):>11.2f}"
              f"{np.mean([stats.kurtosis(r['dec']) for r in grupo]):>9.2f}")

    if "0_normal" in faixas:
        n_min, n_max = faixas["0_normal"]
        sobrepoem = [p for p, (a, b) in faixas.items()
                     if p != "0_normal" and a <= n_max and b >= n_min]
        if sobrepoem:
            print()
            print(f"A faixa de RMS a {alvo} Hz da classe normal se sobrepoe a de: "
                  f"{', '.join(sobrepoem)}.")
            print("   Amplitude sozinha nao separa as classes na taxa do sensor.")
    return faixas


def espectro(registros):
    """Fracao da energia por banda no sinal original."""
    titulo("5. ESPECTRO - onde mora a energia do sinal original")

    nyquist = B.TARGET_SAMPLE_RATE_HZ // 2
    bandas = [(0, nyquist), (nyquist, 2000), (2000, 5000), (5000, np.inf)]
    rotulos = [f"0-{nyquist}", f"{nyquist}-2k", "2k-5k", ">5k"]

    print(f"{'arquivo':<24}" + "".join(r.rjust(11) for r in rotulos) +
          "     descartado")
    print("-" * 78)

    descartes = {}
    for reg in registros:
        f, pxx = signal.welch(reg["bruto"], fs=reg["fs"], nperseg=8192)
        total = pxx.sum()
        fracoes = [pxx[(f >= a) & (f < b)].sum() / total for a, b in bandas]
        # Tudo acima do Nyquist de destino e removido pelo filtro anti-aliasing.
        perdido = sum(fracoes[1:])
        descartes.setdefault(reg["pasta"], []).append(perdido)
        print(f"{reg['pasta'] + '/' + reg['nome']:<24}" +
              "".join(f"{100 * v:>10.1f}%" for v in fracoes) +
              f"{100 * perdido:>14.1f}%")

    print("-" * 78)
    print(f"Energia descartada pela decimacao para {B.TARGET_SAMPLE_RATE_HZ} Hz, "
          f"por classe:")
    for pasta, valores in descartes.items():
        print(f"   {pasta:<16} media {100 * np.mean(valores):>5.1f}%  "
              f"(min {100 * min(valores):>5.1f}%, max {100 * max(valores):>5.1f}%)")
    return descartes


def razao_envelope(x, fs, alvo, banda=None):
    """Razao pico/mediana do espectro de envelope em torno de `alvo`.

    O defeito de rolamento nao aparece como um tom em `alvo`: aparece como uma
    sequencia de impulsos que MODULA uma ressonancia estrutural de alta
    frequencia. Por isso a medida certa e a demodulacao (envelope de Hilbert), e
    nao a FFT direta do sinal.
    """
    if banda is not None:
        alta = min(0.99, banda[1] / (fs / 2))
        b, a = signal.butter(4, [banda[0] / (fs / 2), alta], btype="band")
        x = signal.filtfilt(b, a, x)

    envelope = np.abs(signal.hilbert(x))
    envelope = envelope - envelope.mean()
    f, pxx = signal.welch(envelope, fs=fs, nperseg=min(len(envelope), 16384))

    util = (f > 5) & (f < 500)
    f, pxx = f[util], pxx[util]
    vizinhanca = np.abs(f - alvo) < 3.0
    if not vizinhanca.any():
        return float("nan")
    return float(pxx[vizinhanca].max() / np.median(pxx))


def assinatura(registros):
    """A frequencia caracteristica do defeito sobrevive a decimacao?"""
    titulo("6. ASSINATURA - BPFI / BPFO no espectro de envelope")

    print("Razao pico/mediana na frequencia caracteristica. Ruido puro fica")
    print(f"proximo de 1-{PISO_ENVELOPE:.0f}; valores acima indicam defeito visivel.")
    print()
    print(f"{'arquivo':<24}{'severidade':<12}{'BPFx(Hz)':>10}"
          f"{'2k-5kHz':>12}{'apos 1kHz':>12}   situacao")
    print("-" * 78)

    mudos = []
    for reg in registros:
        if reg["rotulo"] == 0:
            continue
        mult = BPFI_MULT if reg["rotulo"] == 1 else BPFO_MULT
        alvo = mult * RPM_NOMINAL[reg["carga"]] / 60.0
        ressonancia = razao_envelope(reg["bruto"], reg["fs"], alvo, banda=(2000, 5000))
        decimado = razao_envelope(reg["dec"], B.TARGET_SAMPLE_RATE_HZ, alvo)

        if ressonancia < PISO_ENVELOPE:
            situacao = "SEM assinatura nem no sinal original"
            mudos.append(reg)
        elif decimado < PISO_ENVELOPE:
            situacao = "assinatura perdida na decimacao"
            mudos.append(reg)
        else:
            atenuacao = ressonancia / decimado
            situacao = ("sobrevive intacta" if atenuacao < 1.5
                        else f"sobrevive, {atenuacao:.0f}x mais fraca")

        print(f"{reg['pasta'] + '/' + reg['nome']:<24}{reg['severidade']:<12}"
              f"{alvo:>10.1f}{ressonancia:>12.1f}{decimado:>12.1f}   {situacao}")

    print("-" * 78)
    if mudos:
        print(f"{len(mudos)} arquivo(s) chegam ao modelo sem assinatura utilizavel:")
        for reg in mudos:
            print(f"   {reg['pasta']}/{reg['nome']}  ({reg['severidade']}, "
                  f"{reg['carga']} hp)")
        print()
        print("   Janelas desses arquivos carregam um rotulo de falha sobre um sinal")
        print("   em que a falha nao e observavel. Sao rotulos corretos na bancada e")
        print("   incorretos do ponto de vista do que o sensor consegue medir.")
    else:
        print("Todos os arquivos de falha mantem assinatura acima do piso de ruido.")
    return mudos


def escala(registros):
    """Quanto sinal independente existe, por classe."""
    titulo("7. ESCALA - sinal independente disponivel")

    alvo = B.TARGET_SAMPLE_RATE_HZ
    total = sum(len(r["dec"]) for r in registros)
    print(f"{len(registros)} arquivos | {total} amostras a {alvo} Hz = "
          f"{total / alvo:.1f} s | {total // B.WINDOW_SIZE} janelas independentes")
    print()
    print(f"{'classe':<16}{'arquivos':>10}{'segundos':>10}{'% do total':>12}"
          f"{'janelas indep.':>16}")
    print("-" * 78)
    for rotulo, pasta in enumerate(CLASSES):
        grupo = [r for r in registros if r["rotulo"] == rotulo]
        if not grupo:
            continue
        amostras = sum(len(r["dec"]) for r in grupo)
        print(f"{pasta:<16}{len(grupo):>10}{amostras / alvo:>10.1f}"
              f"{100 * amostras / total:>11.1f}%"
              f"{amostras // B.WINDOW_SIZE:>16}")

    print("-" * 78)
    print(f"O build.py usa sobreposicao de {100 * B.WINDOW_OVERLAP:.2f}%, o que "
          f"multiplica a contagem de")
    print("janelas sem criar informacao nova. A coluna acima e o que existe de fato.")


def main():
    print("Auditoria do dataset CWRU")
    print(f"Taxa de destino: {B.TARGET_SAMPLE_RATE_HZ} Hz "
          f"(Nyquist = {B.TARGET_SAMPLE_RATE_HZ // 2} Hz) | "
          f"janela = {B.WINDOW_SIZE} amostras")

    ausentes = inventario()
    registros = carregar_tudo()
    if not registros:
        print("\nNenhum .mat encontrado em data/. Execute 'python download_cwru.py'.")
        return 1

    divergentes, duplicados = integridade(registros)
    buracos = cobertura(registros)
    caracterizacao(registros)
    espectro(registros)
    mudos = assinatura(registros)
    escala(registros)

    titulo("RESUMO")
    achados = [
        (len(ausentes), "arquivo(s) do catalogo ausentes em data/"),
        (len(divergentes), "arquivo(s) entregando a serie de outro ensaio"),
        (len(duplicados), "grupo(s) de arquivos com conteudo identico"),
        (buracos, "celula(s) vazias no plano classe x severidade x carga"),
        (len(mudos), "arquivo(s) de falha sem assinatura utilizavel a "
                     f"{B.TARGET_SAMPLE_RATE_HZ} Hz"),
    ]
    for quantidade, descricao in achados:
        marca = "ok  " if quantidade == 0 else "AVISO"
        print(f"  [{marca}] {quantidade:>2}  {descricao}")

    print()
    print("Interpretacao e plano de acao: docs/dataset.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
