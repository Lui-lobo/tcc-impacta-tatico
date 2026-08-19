"""Catalogo mestre dos ensaios e parametros da curadoria.

Este modulo e a UNICA fonte de verdade sobre o que cada arquivo .mat contem.
Antes dele, a mesma informacao estava duplicada em tres lugares (o CATALOGO do
download_cwru.py, o CARGA_POR_ARQUIVO do validacao_por_carga.py e o
SOURCE_RATE_BY_FILE do build.py), com o risco obvio de divergirem.

O nome do arquivo no CWRU e apenas um numero de catalogo: 106.mat nao diz que e
pista interna, 0,007", 1 hp. Essa traducao vive aqui.
"""
import os

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_BRUTO = os.path.join(RAIZ, "data")
DIR_CURADO = os.path.join(RAIZ, "data_curado")
DIR_RELATORIOS = os.path.join(RAIZ, "dataset_engineering", "relatorios")
DIR_FIGURAS = os.path.join(DIR_RELATORIOS, "figuras")
MANIFESTO = os.path.join(DIR_CURADO, "manifesto.json")

# ---------------------------------------------------------------------------
# Amostragem
# ---------------------------------------------------------------------------
# Teto real do MPU6050. Toda a analise deste fluxo existe para medir o que
# acontece com o sinal ao ser trazido de 12/48 kHz para ca.
TAXA_ALVO_HZ = 1000
NYQUIST_ALVO_HZ = TAXA_ALVO_HZ // 2
TAMANHO_JANELA = 512

TAXA_NORMAL_HZ = 48000   # ensaios "Normal Baseline" (97-100)
TAXA_FALHA_HZ = 12000    # ensaios "12k Drive End Bearing Fault Data"

# ---------------------------------------------------------------------------
# Geometria do rolamento SKF 6205-2RS JEM (lado do acionamento)
# ---------------------------------------------------------------------------
# Multiplicadores da rotacao do eixo (em rev/s) que dao a frequencia de
# passagem de esfera sobre o defeito.
BPFI_MULT = 5.4152   # Ball Pass Frequency, Inner race
BPFO_MULT = 3.5848   # Ball Pass Frequency, Outer race

# Banda de ressonancia estrutural do mancal. E ela que os impactos do defeito
# excitam, e e nela que a demodulacao classica procura a assinatura.
BANDA_RESSONANCIA_HZ = (2000, 5000)

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASSES = {0: "0_normal", 1: "1_inner_race", 2: "2_outer_race"}
NOMES_CURTOS = {0: "normal", 1: "pista interna", 2: "pista externa"}

RPM_NOMINAL = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# ---------------------------------------------------------------------------
# Catalogo: (numero, classe, severidade, carga_hp)
# ---------------------------------------------------------------------------
_TABELA = [
    # Normal Baseline, 48 kHz. Sao os unicos 4 ensaios de rolamento saudavel
    # que o CWRU publica - o desbalanceamento das classes comeca aqui.
    (97, 0, None, 0), (98, 0, None, 1), (99, 0, None, 2), (100, 0, None, 3),

    # Pista interna, 12 kHz Drive End.
    (105, 1, '0,007"', 0), (106, 1, '0,007"', 1),
    (107, 1, '0,007"', 2), (108, 1, '0,007"', 3),
    (169, 1, '0,014"', 0), (170, 1, '0,014"', 1),
    (171, 1, '0,014"', 2), (172, 1, '0,014"', 3),
    (209, 1, '0,021"', 0), (210, 1, '0,021"', 1),
    (211, 1, '0,021"', 2), (212, 1, '0,021"', 3),

    # Pista externa, 12 kHz Drive End, defeito centrado em 6 horas (posicao de
    # maxima carga - a mais usada na literatura).
    (130, 2, '0,007"', 0), (131, 2, '0,007"', 1),
    (132, 2, '0,007"', 2), (133, 2, '0,007"', 3),
    (197, 2, '0,014"', 0), (198, 2, '0,014"', 1),
    (199, 2, '0,014"', 2), (200, 2, '0,014"', 3),
    (234, 2, '0,021"', 0), (235, 2, '0,021"', 1),
    (236, 2, '0,021"', 2), (237, 2, '0,021"', 3),
]


class Ensaio:
    """Um arquivo .mat do CWRU e tudo que se sabe sobre ele a priori."""

    __slots__ = ("numero", "classe", "severidade", "carga")

    def __init__(self, numero, classe, severidade, carga):
        self.numero = numero
        self.classe = classe
        self.severidade = severidade
        self.carga = carga

    @property
    def pasta(self):
        return CLASSES[self.classe]

    @property
    def arquivo(self):
        return f"{self.numero}.mat"

    @property
    def caminho(self):
        return os.path.join(DIR_BRUTO, self.pasta, self.arquivo)

    @property
    def taxa_origem(self):
        return TAXA_NORMAL_HZ if self.classe == 0 else TAXA_FALHA_HZ

    @property
    def chave_mat(self):
        """Nome exato da variavel do canal Drive End dentro do .mat.

        A correcao T1 depende disto: alguns arquivos do CWRU carregam as
        variaveis de MAIS DE UM ensaio, e escolher "a primeira chave que contem
        _DE_time" entrega a serie errada. Derivar a chave do numero do arquivo
        elimina a ambiguidade.
        """
        return f"X{self.numero:03d}_DE_time"

    @property
    def rpm(self):
        return RPM_NOMINAL[self.carga]

    @property
    def frequencia_defeito(self):
        """BPFI ou BPFO em Hz. Zero para a classe normal (nao ha defeito)."""
        if self.classe == 0:
            return 0.0
        mult = BPFI_MULT if self.classe == 1 else BPFO_MULT
        return mult * self.rpm / 60.0

    @property
    def rotulo(self):
        sev = self.severidade or "-"
        return f"{self.numero}.mat ({NOMES_CURTOS[self.classe]}, {sev}, {self.carga} hp)"

    def __repr__(self):
        return f"<Ensaio {self.numero} classe={self.classe} carga={self.carga}>"


ENSAIOS = [Ensaio(*linha) for linha in _TABELA]
POR_NUMERO = {e.numero: e for e in ENSAIOS}

SEVERIDADES = ['0,007"', '0,014"', '0,021"']
CARGAS = sorted(RPM_NOMINAL)

# ---------------------------------------------------------------------------
# Criterios de curadoria
# ---------------------------------------------------------------------------
# Razao pico/mediana do espectro de envelope na frequencia caracteristica.
# Abaixo deste valor o pico nao se distingue do fundo de ruido, e o ensaio nao
# carrega assinatura de defeito observavel - nem no sinal original.
PISO_ENVELOPE = 10.0

# Sobreposicao usada pelo build.py. Reproduzida aqui so para converter contagem
# de janelas sobrepostas em janelas independentes nos relatorios.
SOBREPOSICAO_TREINO = 0.9375

# Semente unica para tudo que sorteia neste fluxo.
SEMENTE = 42


def garantir_diretorios():
    for caminho in (DIR_RELATORIOS, DIR_FIGURAS):
        os.makedirs(caminho, exist_ok=True)
