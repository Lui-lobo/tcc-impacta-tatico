"""Etapa 1 - Leitura e demonstracao do dataset bruto.

Responde a pergunta mais basica e a que nunca tinha sido feita neste projeto:
o que exatamente ha nesses arquivos?

Produz um inventario textual e quatro figuras que MOSTRAM o sinal, em vez de
apenas descreve-lo. As figuras vao para dataset_engineering/relatorios/figuras/
e servem diretamente ao texto do TCC.

    python -m dataset_engineering.etapa1_leitura
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from dataset_engineering import config, leitura

CORES = {0: "#2E7D32", 1: "#C62828", 2: "#1565C0"}


def cabecalho(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Inventario
# ---------------------------------------------------------------------------
def inventario(registros, ausentes):
    cabecalho("1.1 INVENTARIO")

    print(f"Catalogados: {len(config.ENSAIOS)} | Presentes: {len(registros)} | "
          f"Ausentes: {len(ausentes)}")
    if ausentes:
        print()
        print("AUSENTES:")
        for e in ausentes:
            print(f"   {e.pasta}/{e.arquivo:<10} {e.rotulo}")
        print("   Execute 'python download_cwru.py' para baixa-los.")

    print()
    print(f"{'arquivo':<12}{'classe':<15}{'sever.':<9}{'carga':>6}"
          f"{'rpm cat':>9}{'rpm .mat':>10}{'fs':>7}{'dur_s':>8}{'canal':>16}")
    print("-" * 78)
    for r in registros:
        e = r["ensaio"]
        rpm_m = r["rpm_medido"]
        divergencia = ""
        if rpm_m is not None and abs(rpm_m - e.rpm) > 30:
            divergencia = "  <<< rpm fora do esperado"
        print(f"{e.arquivo:<12}{config.NOMES_CURTOS[e.classe]:<15}"
              f"{(e.severidade or '-'):<9}{e.carga:>4} hp{e.rpm:>9}"
              f"{(rpm_m if rpm_m is not None else '-'):>10}{e.taxa_origem:>7}"
              f"{len(r['bruto']) / e.taxa_origem:>8.2f}"
              f"{r['chave_usada']:>16}{divergencia}")


def canais_disponiveis(registros):
    """Quais canais cada .mat traz. O projeto usa so o DE; os outros existem."""
    cabecalho("1.2 CANAIS PRESENTES NOS ARQUIVOS")

    print("O projeto usa apenas o canal DE (drive end), por ser o mais proximo")
    print("do defeito e o que corresponde a posicao do MPU6050 na bancada.")
    print()
    print(f"{'arquivo':<12}{'DE':>5}{'FE':>5}{'BA':>5}{'RPM':>6}"
          f"   ensaios dentro do arquivo")
    print("-" * 78)
    for r in registros:
        e = r["ensaio"]
        chaves = leitura.chaves_de(e.caminho)
        tem = lambda suf: "sim" if any(k.endswith(suf) for k in chaves) else "-"
        n = r["ensaios_no_arquivo"]
        aviso = "  <<< mais de um ensaio no mesmo .mat" if n > 1 else ""
        print(f"{e.arquivo:<12}{tem('_DE_time'):>5}{tem('_FE_time'):>5}"
              f"{tem('_BA_time'):>5}{tem('RPM'):>6}   {n}{aviso}")


def cobertura(registros):
    cabecalho("1.3 COBERTURA DO PLANO FATORIAL")

    presentes = {(r["ensaio"].classe, r["ensaio"].severidade, r["ensaio"].carga)
                 for r in registros}
    print(f"{'classe':<16}{'severidade':<12}" +
          "".join(f"{c} hp".rjust(10) for c in config.CARGAS))
    print("-" * 78)

    vazias = 0
    for classe, pasta in config.CLASSES.items():
        severidades = ([None] if classe == 0 else config.SEVERIDADES)
        for sev in severidades:
            celulas = []
            for carga in config.CARGAS:
                ok = (classe, sev, carga) in presentes
                vazias += 0 if ok else 1
                celulas.append(("sim" if ok else "AUSENTE").rjust(10))
            print(f"{pasta:<16}{(sev or '-'):<12}" + "".join(celulas))

    print("-" * 78)
    print(f"Celulas vazias: {vazias}")
    return vazias


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _exemplar(registros, classe, carga=3, severidade='0,021"'):
    """Um ensaio representativo da classe, para as figuras."""
    for r in registros:
        e = r["ensaio"]
        if e.classe == classe and e.carga == carga and (
                classe == 0 or e.severidade == severidade):
            return r
    candidatos = [r for r in registros if r["ensaio"].classe == classe]
    return candidatos[0] if candidatos else None


def figura_formas_de_onda(registros, caminho):
    """O sinal, antes e depois da decimacao. A figura mais direta do problema."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    duracao = 0.5  # segundos mostrados

    for i, (classe, pasta) in enumerate(config.CLASSES.items()):
        r = _exemplar(registros, classe)
        if r is None:
            continue
        e = r["ensaio"]

        n_o = int(duracao * e.taxa_origem)
        t_o = np.arange(n_o) / e.taxa_origem
        axes[i][0].plot(t_o, r["bruto"][:n_o], lw=0.4, color=CORES[classe])
        axes[i][0].set_ylabel(f"{config.NOMES_CURTOS[classe]}\n[g]", fontsize=10)
        axes[i][0].grid(alpha=0.3)

        n_d = int(duracao * config.TAXA_ALVO_HZ)
        t_d = np.arange(n_d) / config.TAXA_ALVO_HZ
        axes[i][1].plot(t_d, r["dec"][:n_d], lw=0.8, color=CORES[classe])
        axes[i][1].grid(alpha=0.3)

        # Mesma escala vertical nos dois lados: sem isso a decimacao pareceria
        # preservar a amplitude, quando na verdade ela some.
        limite = 1.05 * np.max(np.abs(r["bruto"][:n_o]))
        axes[i][0].set_ylim(-limite, limite)
        axes[i][1].set_ylim(-limite, limite)

    axes[0][0].set_title(f"Original ({config.TAXA_FALHA_HZ // 1000}/"
                         f"{config.TAXA_NORMAL_HZ // 1000} kHz)", fontsize=12)
    axes[0][1].set_title(f"Apos decimacao ({config.TAXA_ALVO_HZ} Hz — "
                         f"o que o ESP32 ve)", fontsize=12)
    for ax in axes[2]:
        ax.set_xlabel("tempo [s]", fontsize=10)
    fig.suptitle("Forma de onda por classe, antes e depois da decimacao",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(caminho, dpi=200)
    plt.close()


def figura_espectros(registros, caminho):
    """Onde mora a energia, e onde cai o corte do anti-aliasing."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for classe, pasta in config.CLASSES.items():
        grupo = [r for r in registros if r["ensaio"].classe == classe]
        if not grupo:
            continue

        acumulado = None
        for r in grupo:
            f, pxx = signal.welch(r["bruto"], fs=r["ensaio"].taxa_origem,
                                  nperseg=8192)
            pxx = pxx / pxx.sum()
            acumulado = pxx if acumulado is None else acumulado + pxx
        axes[0].semilogy(f, acumulado / len(grupo), lw=1.0,
                         color=CORES[classe], label=config.NOMES_CURTOS[classe])

        acumulado = None
        for r in grupo:
            f, pxx = signal.welch(r["dec"], fs=config.TAXA_ALVO_HZ, nperseg=512)
            pxx = pxx / pxx.sum()
            acumulado = pxx if acumulado is None else acumulado + pxx
        axes[1].semilogy(f, acumulado / len(grupo), lw=1.2,
                         color=CORES[classe], label=config.NOMES_CURTOS[classe])

    # Recorta em Nyquist dos arquivos de falha. A classe normal foi gravada a
    # 48 kHz e sozinha esticaria o eixo ate 24 kHz, achatando justamente a
    # faixa que interessa.
    axes[0].set_xlim(0, config.TAXA_FALHA_HZ / 2)
    axes[0].axvline(config.NYQUIST_ALVO_HZ, color="k", ls="--", lw=1.2)
    axes[0].axvspan(config.NYQUIST_ALVO_HZ, config.TAXA_FALHA_HZ / 2,
                    color="k", alpha=0.07)
    axes[0].annotate("descartado pela decimacao",
                     xy=(config.NYQUIST_ALVO_HZ * 1.6,
                         axes[0].get_ylim()[1] * 0.15),
                     fontsize=10)
    axes[0].axvspan(*config.BANDA_RESSONANCIA_HZ, color="orange", alpha=0.15)
    axes[0].set_title("Espectro medio — sinal original", fontsize=12)
    axes[0].set_xlabel("frequencia [Hz]")
    axes[0].set_ylabel("densidade espectral normalizada")

    axes[1].set_title(f"Espectro medio — apos decimar a "
                      f"{config.TAXA_ALVO_HZ} Hz", fontsize=12)
    axes[1].set_xlabel("frequencia [Hz]")

    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=10)

    fig.suptitle("A energia do defeito mora na ressonancia de 2–5 kHz "
                 "(faixa laranja), fora da banda do MPU6050", fontsize=13)
    plt.tight_layout()
    plt.savefig(caminho, dpi=200)
    plt.close()


def figura_envelope(registros, caminho):
    """A assinatura do defeito sobrevive a decimacao? Espectro de envelope."""
    exemplares = [r for r in (_exemplar(registros, 1), _exemplar(registros, 2))
                  if r is not None]
    if not exemplares:
        return

    fig, axes = plt.subplots(len(exemplares), 2, figsize=(14, 4 * len(exemplares)),
                             squeeze=False)

    for i, r in enumerate(exemplares):
        e = r["ensaio"]
        alvo = e.frequencia_defeito

        for j, (x, fs, banda, titulo) in enumerate((
            (r["bruto"], e.taxa_origem, config.BANDA_RESSONANCIA_HZ,
             "original, demodulado em 2–5 kHz"),
            (r["dec"], config.TAXA_ALVO_HZ, None,
             f"apos decimar a {config.TAXA_ALVO_HZ} Hz"),
        )):
            sinal = x
            if banda is not None:
                b, a = signal.butter(
                    4, [banda[0] / (fs / 2), min(0.99, banda[1] / (fs / 2))],
                    btype="band")
                sinal = signal.filtfilt(b, a, sinal)
            env = np.abs(signal.hilbert(sinal))
            env = env - env.mean()
            f, pxx = signal.welch(env, fs=fs, nperseg=min(len(env), 16384))
            util = (f > 5) & (f < config.NYQUIST_ALVO_HZ)

            ax = axes[i][j]
            ax.semilogy(f[util], pxx[util], lw=1.0, color=CORES[e.classe])
            ax.axvline(alvo, color="k", ls="--", lw=1.2)
            for h in (2, 3):
                if alvo * h < config.NYQUIST_ALVO_HZ:
                    ax.axvline(alvo * h, color="k", ls=":", lw=0.8, alpha=0.6)
            ax.set_title(f"{e.rotulo}\n{titulo}", fontsize=10)
            ax.set_xlabel("frequencia [Hz]")
            ax.grid(alpha=0.3, which="both")
            if j == 0:
                ax.set_ylabel("envelope [densidade]")
            ax.annotate(f"{'BPFI' if e.classe == 1 else 'BPFO'} = {alvo:.0f} Hz",
                        xy=(alvo, ax.get_ylim()[1] * 0.3), fontsize=9,
                        xytext=(alvo + 40, ax.get_ylim()[1] * 0.3))

    fig.suptitle("Espectro de envelope: o pico na frequencia caracteristica "
                 "sobrevive a decimacao, mas com margem muito menor", fontsize=13)
    plt.tight_layout()
    plt.savefig(caminho, dpi=200)
    plt.close()


def figura_separabilidade(registros, caminho):
    """RMS x curtose, por janela independente, antes e depois da decimacao."""
    from dataset_engineering.metricas import estatisticas_tempo

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for classe in config.CLASSES:
        grupo = [r for r in registros if r["ensaio"].classe == classe]
        for j, chave in enumerate(("bruto", "dec")):
            xs, ys = [], []
            for r in grupo:
                serie = r[chave]
                fs = (r["ensaio"].taxa_origem if chave == "bruto"
                      else config.TAXA_ALVO_HZ)
                # Janelas de mesma DURACAO nos dois lados, para que a curtose
                # seja calculada sobre o mesmo intervalo fisico.
                tamanho = int(config.TAMANHO_JANELA * fs / config.TAXA_ALVO_HZ)
                jan = leitura.janelas_independentes(serie, tamanho)
                for w in jan:
                    est = estatisticas_tempo(w)
                    xs.append(est["rms"])
                    ys.append(est["curtose"])
            axes[j].scatter(xs, ys, s=6, alpha=0.35, color=CORES[classe],
                            label=config.NOMES_CURTOS[classe])

    axes[0].set_title("Sinal original", fontsize=12)
    axes[1].set_title(f"Apos decimar a {config.TAXA_ALVO_HZ} Hz", fontsize=12)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("RMS [g]")
        ax.set_ylabel("curtose")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, markerscale=2)

    fig.suptitle("Separabilidade por atributos classicos: as nuvens se fundem "
                 "depois da decimacao", fontsize=13)
    plt.tight_layout()
    plt.savefig(caminho, dpi=200)
    plt.close()


def gerar_figuras(registros):
    cabecalho("1.4 FIGURAS")
    config.garantir_diretorios()

    tarefas = [
        ("formas_de_onda.png", figura_formas_de_onda),
        ("espectros.png", figura_espectros),
        ("envelope.png", figura_envelope),
        ("separabilidade.png", figura_separabilidade),
    ]
    gerados = []
    for nome, funcao in tarefas:
        caminho = os.path.join(config.DIR_FIGURAS, nome)
        funcao(registros, caminho)
        gerados.append(caminho)
        print(f"   {os.path.relpath(caminho, config.RAIZ)}")
    return gerados


# ---------------------------------------------------------------------------
def executar(verbose=True):
    print("ETAPA 1 - Leitura e demonstracao do dataset bruto")
    print(f"Origem: {os.path.relpath(config.DIR_BRUTO, config.RAIZ)} | "
          f"taxa de destino: {config.TAXA_ALVO_HZ} Hz "
          f"(Nyquist = {config.NYQUIST_ALVO_HZ} Hz)")

    # estrito=False: a etapa 1 descreve o dataset como ele e, inclusive quando
    # um arquivo nao tem a chave esperada. Recusar so faz sentido na etapa 3.
    registros, ausentes = leitura.carregar_ensaios(estrito=False)
    if not registros:
        print("\nNenhum .mat encontrado. Execute 'python download_cwru.py'.")
        return None

    inventario(registros, ausentes)
    canais_disponiveis(registros)
    vazias = cobertura(registros)
    figuras = gerar_figuras(registros)

    return {
        "registros": registros,
        "ausentes": ausentes,
        "celulas_vazias": vazias,
        "figuras": figuras,
    }


if __name__ == "__main__":
    executar()
