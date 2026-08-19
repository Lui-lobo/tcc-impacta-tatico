"""Exporta amostras do dataset em CSV, antes e depois da curadoria.

MOTIVACAO
---------
O .mat do CWRU e um formato binario do MATLAB: nao abre em editor de texto, nao
abre em planilha, e nem sempre contem o que o nome do arquivo promete. Isso
torna o dataset dificil de MOSTRAR - e um trabalho academico precisa que o
leitor consiga inspecionar o dado, nao so acreditar na descricao dele.

Este modulo produz cinco CSVs pequenos, cada um respondendo a uma pergunta:

  01_catalogo_bruto.csv    O que existe no dataset como ele vem do CWRU?
  02_catalogo_curado.csv   O que sobrou depois da curadoria, e por que?
  03_sinal_bruto.csv       Como e o sinal na taxa original de gravacao?
  04_janela_modelo.csv     Como e a janela que efetivamente entra na rede?
  05_correcao_t1.csv       Que diferenca fez corrigir a leitura de 99.mat?

Sao arquivos versionados no repositorio, de proposito: eles descrevem o dado e
sao pequenos o bastante para isso. As series completas continuam fora do
controle de versao.

USO
---
    python -m dataset_engineering.exportar_amostras

Roda tambem no fim de `python -m dataset_engineering.executar`.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.io

from dataset_engineering import config, leitura, metricas

DIR_AMOSTRAS = os.path.join(config.RAIZ, "dataset_engineering", "amostras")

# Trecho exportado nos CSVs de sinal. 100 ms cobre cerca de 10 passagens de
# esfera em BPFO (~105 Hz) e 16 em BPFI (~160 Hz) - o suficiente para que a
# periodicidade dos impactos apareca a olho num grafico de planilha.
DURACAO_AMOSTRA_S = 0.100

# Deslocamento a partir do inicio da gravacao. Evita o transitorio de partida.
OFFSET_S = 2.0

# Um ensaio por classe, todos na MESMA carga e na MESMA severidade, para que a
# comparacao entre eles isole o tipo de defeito. 0,021" e a severidade em que a
# assinatura e mais visivel.
REPRESENTANTES = {0: 100, 1: 212, 2: 237}


def _fmt(valor, casas=6):
    """Formata um valor para a celula do CSV.

    Decimal com virgula: e o que o Excel em portugues espera, e com o separador
    de campo em ';' nao ha ambiguidade. Quem ler em Python passa
    `sep=';', decimal=','`.
    """
    if valor is None:
        return ""
    if isinstance(valor, bool):
        return "sim" if valor else "nao"
    if isinstance(valor, float):
        if np.isnan(valor):
            return ""
        return f"{valor:.{casas}f}".replace(".", ",")
    return str(valor)


def _severidade(valor):
    '''Troca a marca de polegada por ' pol': 0,014" -> 0,014 pol.

    A aspa dupla obrigaria o escritor de CSV a envolver o campo em aspas e a
    duplicar a original. O resultado e um CSV correto, mas com celulas que um
    editor de texto mostra cheias de aspas seguidas - e o ponto destes arquivos
    e justamente serem lidos por pessoas.
    '''
    return (valor or "").replace('"', " pol").strip()


def _escrever(caminho, colunas, linhas):
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow(colunas)
        escritor.writerows(linhas)
    tamanho = os.path.getsize(caminho) / 1024.0
    print(f"   {os.path.relpath(caminho, config.RAIZ):<52} "
          f"{len(linhas):>7} linhas  {tamanho:>7.1f} KB")
    return caminho


# ---------------------------------------------------------------------------
# 01 - Catalogo bruto
# ---------------------------------------------------------------------------
def catalogo_bruto():
    """Um registro por ensaio, lido como o build.py lia antes da curadoria."""
    colunas = [
        "numero", "arquivo", "classe", "classe_nome", "severidade", "carga_hp",
        "rpm_nominal", "rpm_medido", "fs_origem_hz", "amostras", "duracao_s",
        "canal_lido", "canal_esperado", "leitura_correta",
        "rms_g", "curtose", "fator_crista", "energia_acima_500hz_pct",
        "envelope_bpfx", "frequencia_defeito_hz",
    ]

    linhas = []
    for ensaio in config.ENSAIOS:
        if not os.path.exists(ensaio.caminho):
            continue
        mat = scipy.io.loadmat(ensaio.caminho)
        # A leitura INGENUA: a primeira chave '_DE_time' que aparecer. E o que
        # o pipeline fazia, e o que este CSV precisa documentar.
        ingenua = next((k for k in mat if "_DE_time" in k), None)
        serie = mat[ingenua].flatten().astype(np.float64)

        tempo = metricas.estatisticas_tempo(serie)
        bandas = metricas.bandas_energia(serie, ensaio.taxa_origem)
        envelope = (metricas.razao_envelope(
            serie, ensaio.taxa_origem, ensaio.frequencia_defeito,
            banda=config.BANDA_RESSONANCIA_HZ) if ensaio.classe else float("nan"))

        linhas.append([
            ensaio.numero, ensaio.arquivo, ensaio.classe,
            config.NOMES_CURTOS[ensaio.classe], _severidade(ensaio.severidade),
            ensaio.carga, ensaio.rpm, _fmt(leitura.rpm_medido(ensaio)),
            ensaio.taxa_origem, len(serie),
            _fmt(len(serie) / ensaio.taxa_origem, 3),
            ingenua, ensaio.chave_mat, _fmt(ingenua == ensaio.chave_mat),
            _fmt(tempo["rms"]), _fmt(tempo["curtose"], 2),
            _fmt(tempo["crista"], 2),
            _fmt(100 * bandas.get("descartado", float("nan")), 1),
            _fmt(envelope, 1), _fmt(ensaio.frequencia_defeito, 1),
        ])
    return _escrever(os.path.join(DIR_AMOSTRAS, "01_catalogo_bruto.csv"),
                     colunas, linhas)


# ---------------------------------------------------------------------------
# 02 - Catalogo curado
# ---------------------------------------------------------------------------
def catalogo_curado():
    """O mesmo catalogo depois da curadoria, com o destino de cada ensaio."""
    if not os.path.exists(config.MANIFESTO):
        print("   [pulado] 02_catalogo_curado.csv - rode a etapa 3 primeiro.")
        return None

    with open(config.MANIFESTO, encoding="utf-8") as f:
        manifesto = json.load(f)

    colunas = [
        "numero", "arquivo_origem", "classe", "classe_nome", "severidade",
        "carga_hp", "rpm_nominal", "fs_origem_hz", "fs_hz", "amostras",
        "duracao_s", "canal_lido", "corrigido_t1", "envelope_ressonancia",
        "status", "motivo", "arquivo_npy",
    ]

    linhas = []
    for e in manifesto["ensaios"]:
        linhas.append([
            e["numero"], e["arquivo_origem"], e["classe"],
            config.NOMES_CURTOS[e["classe"]], _severidade(e["severidade"]),
            e["carga_hp"], e["rpm_nominal"], e["fs_origem_hz"], e["fs_hz"],
            e["amostras"], _fmt(e["duracao_s"], 3), e["chave_mat"],
            _fmt(e["corrigido_t1"]), _fmt(e["envelope_ressonancia"], 1),
            e["status"], e["motivo"] or "", e["npy"] or "",
        ])
    return _escrever(os.path.join(DIR_AMOSTRAS, "02_catalogo_curado.csv"),
                     colunas, linhas)


# ---------------------------------------------------------------------------
# 03 / 04 - Sinal
# ---------------------------------------------------------------------------
def _representantes():
    """Le os tres ensaios representativos, bruto e decimado."""
    escolhidos = [config.POR_NUMERO[n] for n in REPRESENTANTES.values()]
    registros, ausentes = leitura.carregar_ensaios(escolhidos, estrito=True)
    if ausentes:
        print(f"   [aviso] ausentes: {[e.arquivo for e in ausentes]}")
    return registros


def sinal_bruto(registros):
    """Trecho na taxa ORIGINAL de gravacao.

    Formato longo (uma linha por amostra por classe) porque as classes tem
    taxas diferentes - normal a 48 kHz, falhas a 12 kHz - e nao compartilham
    base de tempo. Em formato largo seria preciso inventar interpolacao.
    """
    colunas = ["classe_nome", "arquivo", "fs_hz", "amostra", "tempo_s",
               "aceleracao_g"]
    linhas = []
    for r in registros:
        e = r["ensaio"]
        fs = e.taxa_origem
        inicio = int(OFFSET_S * fs)
        n = int(DURACAO_AMOSTRA_S * fs)
        trecho = r["bruto"][inicio:inicio + n]
        for i, v in enumerate(trecho):
            linhas.append([config.NOMES_CURTOS[e.classe], e.arquivo, fs, i,
                           _fmt(i / fs, 7), _fmt(v)])
    return _escrever(os.path.join(DIR_AMOSTRAS, "03_sinal_bruto.csv"),
                     colunas, linhas)


def janela_modelo(registros):
    """A janela de 512 amostras a 1 kHz - exatamente o que entra na rede.

    Aqui o formato largo funciona: depois da decimacao as tres classes
    compartilham a mesma base de tempo, e por em colunas lado a lado permite
    plotar as tres numa planilha sem nenhum tratamento.
    """
    colunas = ["amostra", "tempo_s"]
    series = {}
    for r in registros:
        e = r["ensaio"]
        inicio = int(OFFSET_S * config.TAXA_ALVO_HZ)
        trecho = r["dec"][inicio:inicio + config.TAMANHO_JANELA]
        series[e.classe] = trecho
        colunas.append(f"{config.NOMES_CURTOS[e.classe].replace(' ', '_')}_g")

    if not series:
        return None

    tamanho = min(len(s) for s in series.values())
    linhas = []
    for i in range(tamanho):
        linha = [i, _fmt(i / config.TAXA_ALVO_HZ, 4)]
        for classe in sorted(series):
            linha.append(_fmt(series[classe][i]))
        linhas.append(linha)

    return _escrever(os.path.join(DIR_AMOSTRAS, "04_janela_modelo.csv"),
                     colunas, linhas)


# ---------------------------------------------------------------------------
# 05 - A correcao T1, lado a lado
# ---------------------------------------------------------------------------
def correcao_t1():
    """As duas series que 99.mat pode entregar, na mesma base de tempo.

    E o CSV que torna a tratativa T1 verificavel por inspecao: se as duas
    colunas fossem iguais, a correcao seria cosmetica. Elas nao sao.
    """
    candidatos = [e for e in config.ENSAIOS
                  if os.path.exists(e.caminho)
                  and len([k for k in scipy.io.loadmat(e.caminho)
                           if "_DE_time" in k]) > 1]
    if not candidatos:
        print("   [pulado] 05_correcao_t1.csv - nenhum .mat com mais de um ensaio.")
        return None

    ensaio = candidatos[0]
    mat = scipy.io.loadmat(ensaio.caminho)
    ingenua = next(k for k in mat if "_DE_time" in k)

    serie_ingenua = leitura.decimar(
        mat[ingenua].flatten().astype(np.float64), ensaio.taxa_origem)
    serie_correta = leitura.decimar(
        mat[ensaio.chave_mat].flatten().astype(np.float64), ensaio.taxa_origem)

    inicio = int(OFFSET_S * config.TAXA_ALVO_HZ)
    n = min(config.TAMANHO_JANELA,
            len(serie_ingenua) - inicio, len(serie_correta) - inicio)

    colunas = ["amostra", "tempo_s",
               f"leitura_ingenua_{ingenua}_g",
               f"leitura_corrigida_{ensaio.chave_mat}_g",
               "diferenca_g"]
    linhas = []
    for i in range(n):
        a = serie_ingenua[inicio + i]
        b = serie_correta[inicio + i]
        linhas.append([i, _fmt(i / config.TAXA_ALVO_HZ, 4),
                       _fmt(a), _fmt(b), _fmt(b - a)])

    caminho = _escrever(os.path.join(DIR_AMOSTRAS, "05_correcao_t1.csv"),
                        colunas, linhas)
    print(f"      {ensaio.arquivo}: '{ingenua}' e '{ensaio.chave_mat}' | "
          f"RMS {np.sqrt(np.mean(serie_ingenua ** 2)):.6f} vs "
          f"{np.sqrt(np.mean(serie_correta ** 2)):.6f} g")
    return caminho


# ---------------------------------------------------------------------------
def executar():
    print("Exportando amostras do dataset em CSV")
    os.makedirs(DIR_AMOSTRAS, exist_ok=True)
    print(f"Destino: {os.path.relpath(DIR_AMOSTRAS, config.RAIZ)}")
    print()

    gerados = [catalogo_bruto(), catalogo_curado()]
    registros = _representantes()
    if registros:
        gerados += [sinal_bruto(registros), janela_modelo(registros)]
    gerados.append(correcao_t1())

    gerados = [g for g in gerados if g]
    print()
    print(f"{len(gerados)} arquivo(s) gerados. Separador: ';' | decimal: ','")
    print("Abrem com duplo clique no Excel/LibreOffice em portugues.")
    print("Em Python:  pd.read_csv(caminho, sep=';', decimal=',')")
    return gerados


if __name__ == "__main__":
    executar()
