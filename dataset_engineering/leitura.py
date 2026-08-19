"""Leitura dos .mat brutos, com a selecao de canal corrigida (tratativa T1).

DIFERENCA PARA O src/pipeline/data_processor.py
-----------------------------------------------
O `DataProcessor.load_mat_file()` escolhe a primeira chave que contem
'_DE_time'. Quando o .mat carrega as variaveis de mais de um ensaio - o caso de
99.mat, que traz tambem as de 98.mat - essa primeira chave e a do ensaio errado,
e o pipeline treina com a serie de outro nivel de carga sem emitir um unico
aviso.

Aqui a chave e derivada do NUMERO do arquivo (config.Ensaio.chave_mat). Se ela
nao existir, a leitura falha em vez de adivinhar: numa auditoria, silencio e
pior que erro.

A decimacao continua vindo do DataProcessor, de proposito. Se este modulo
reimplementasse o filtro anti-aliasing, o dataset curado deixaria de ser
comparavel ao que o build.py produz.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.io

from src.pipeline.data_processor import DataProcessor
from dataset_engineering import config

_PROCESSADOR = DataProcessor(window_size=config.TAMANHO_JANELA,
                             overlap=config.SOBREPOSICAO_TREINO)


class LeituraDivergente(Exception):
    """A chave esperada nao existe no arquivo."""


def chaves_de(caminho):
    """Todas as variaveis de sinal presentes no .mat, na ordem do arquivo."""
    mat = scipy.io.loadmat(caminho)
    return [k for k in mat if not k.startswith("__")]


def ler_bruto(ensaio, estrito=True):
    """Devolve (serie, diagnostico) do canal Drive End do ensaio.

    O diagnostico registra qual chave o leitor ingenuo teria escolhido, para que
    a etapa 2 possa quantificar o impacto da correcao em vez de apenas afirmar
    que ela era necessaria.
    """
    if not os.path.exists(ensaio.caminho):
        raise FileNotFoundError(ensaio.caminho)

    mat = scipy.io.loadmat(ensaio.caminho)
    disponiveis = [k for k in mat if "_DE_time" in k]
    ingenua = disponiveis[0] if disponiveis else None
    correta = ensaio.chave_mat

    if correta not in mat:
        if estrito:
            raise LeituraDivergente(
                f"{ensaio.arquivo}: chave '{correta}' ausente. "
                f"Presentes: {disponiveis}")
        correta = ingenua

    diagnostico = {
        "chave_usada": correta,
        "chave_ingenua": ingenua,
        "chaves_de_time": disponiveis,
        # `divergente` e uma propriedade do ARQUIVO: a leitura ingenua pegaria
        # outra serie. `leitura_incorreta` e uma propriedade desta LEITURA:
        # a serie devolvida nao e a do ensaio. As duas so coincidem quando o
        # leitor ingenuo esta em uso - a ablacao da etapa 4 depende de
        # distingui-las.
        "divergente": ingenua != correta,
        "leitura_incorreta": correta != ensaio.chave_mat,
        "ensaios_no_arquivo": len(disponiveis),
    }
    return mat[correta].flatten().astype(np.float64), diagnostico


def rpm_medido(ensaio):
    """Rotacao gravada dentro do .mat, quando existe.

    Nem todo arquivo do CWRU traz a variavel X###RPM. Onde ela existe, serve
    para conferir se o mapa de cargas do catalogo esta correto.
    """
    mat = scipy.io.loadmat(ensaio.caminho)
    chave = f"X{ensaio.numero:03d}RPM"
    if chave in mat:
        return int(np.ravel(mat[chave])[0])
    generica = [k for k in mat if k.endswith("RPM")]
    if generica:
        return int(np.ravel(mat[generica[0]])[0])
    return None


def decimar(serie, taxa_origem, taxa_alvo=None):
    """Reduz a taxa de amostragem com o mesmo filtro que o build.py usa."""
    taxa_alvo = taxa_alvo or config.TAXA_ALVO_HZ
    return _PROCESSADOR.resample_to(serie, taxa_origem, taxa_alvo)


def janelas_independentes(serie, tamanho=None):
    """Fatia a serie em janelas SEM sobreposicao.

    Toda medida deste fluxo usa janelas independentes. A sobreposicao de 93,75%
    do treino multiplica a contagem por 16 sem criar informacao nova; medir
    separabilidade sobre janelas sobrepostas produziria um numero inflado.
    """
    tamanho = tamanho or config.TAMANHO_JANELA
    total = len(serie) // tamanho
    if total == 0:
        return np.empty((0, tamanho))
    return serie[:total * tamanho].reshape(total, tamanho)


def carregar_ensaios(ensaios=None, estrito=True, verbose=False):
    """Le, decima e devolve um registro por ensaio presente em disco.

    Ensaios ausentes sao pulados e listados no campo 'ausentes' do retorno, em
    vez de interromperem a leitura: a etapa 1 precisa conseguir descrever um
    dataset incompleto para que a etapa 3 possa recusa-lo.
    """
    ensaios = ensaios or config.ENSAIOS
    registros, ausentes = [], []

    for ensaio in ensaios:
        if not os.path.exists(ensaio.caminho):
            ausentes.append(ensaio)
            continue
        bruto, diagnostico = ler_bruto(ensaio, estrito=estrito)
        registros.append({
            "ensaio": ensaio,
            "bruto": bruto,
            "dec": decimar(bruto, ensaio.taxa_origem),
            "rpm_medido": rpm_medido(ensaio),
            **diagnostico,
        })
        if verbose:
            print(f"   lido {ensaio.rotulo}")

    return registros, ausentes
