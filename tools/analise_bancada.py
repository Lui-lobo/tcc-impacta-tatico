"""Analisa capturas do laco de inferencia do ESP32 (ensaios de bancada).

MOTIVACAO
---------
O console do firmware imprime, por janela, um bloco com estatisticas de
amostragem, dos tres eixos, da quantizacao e da saida do modelo. Lido a olho,
esse despejo diz pouco; tabulado ao longo de dezenas de janelas, ele mostra
como o sistema se comporta com um sinal que NAO e de rolamento.

E esse o valor de um ensaio de bancada. Nao ha rolamento sob o sensor, entao a
classe predita nao tem significado fisico - e justamente por isso o ensaio
consegue medir o que a acuracia nunca mede: o que o sistema faz quando a
entrada esta fora da distribuicao de treino.

O QUE E EXTRAIDO
----------------
aquisicao   - fs real, jitter, amostras repetidas, erros de I2C. Reconfirma o
              ensaio V1 do protocolo sob vibracao real e sustentada.
amplitude   - desvio padrao do eixo alimentado ao modelo, em multiplos do
              desvio de treino. Mede o quao fora da faixa o sinal esta.
ceifamento  - quantas amostras o quantizador int8 achatou. Acima de ~10% a
              janela chega deformada e a predicao perde qualquer valor.
saida       - classe e confianca. A pergunta central e se o modelo tem algum
              estado de "nao sei" - ou se ele sempre responde com confianca
              maxima, inclusive sobre ruido.
calibracao  - modulo do vetor de gravidade em repouso. Deveria ser 1,000 g.

USO
---
    python tools/analise_bancada.py relatorios/bancada/*.txt
    python tools/analise_bancada.py relatorios/bancada/vibracao_continua.txt

Gera tambem uma figura por captura em relatorios/bancada/.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Acima deste ceifamento a janela chega deformada ao modelo.
CEIFAMENTO_CRITICO = 0.10

# Faixa de amplitude em que o quantizador foi calibrado. Fora dela o modelo
# opera sobre um sinal que nao se parece com nada que ele viu.
FAIXA_CONFIAVEL = (0.5, 2.0)

CORES_CLASSE = {
    "0_normal": "#2E7D32",
    "1_inner_race": "#C62828",
    "2_outer_race": "#1565C0",
}

_PADROES = {
    "janela": re.compile(r"^--- Janela #(\d+) ---"),
    "amostra": re.compile(
        r"fs_real=([\d.]+) Hz \| dur=([\d.]+) ms \| jitter_max=(\d+) us \| "
        r"atrasos=(\d+)"),
    "sensor": re.compile(
        r"repetidas=(\d+)/(\d+).*?\| saturadas=(\d+) \| erros_i2c=(\d+)"),
    # O eixo X vem prefixado por '[EIXOS g]' e os outros dois so indentados,
    # entao a busca nao pode ancorar no inicio da linha.
    "eixo": re.compile(
        r"([XYZ]) med=([-+][\d.]+) dp=([\d.]+) pico=([-+][\d.]+)/([-+][\d.]+)"),
    "entrada": re.compile(
        r"eixo=(\w+) \| dc_removido=(\w+) \| dp=([\d.]+) g \| "
        r"dp_treino=([\d.]+) g \| razao=([\d.]+)x"),
    "quant": re.compile(
        r"int8 min=(-?\d+) max=(-?\d+) \| niveis_distintos=(\d+)/256 \| "
        r"saturados=(\d+)"),
    "predicao": re.compile(r"^=> (\w+) \(([\d.]+)%\) em (\d+) us"),
}


def ler_captura(caminho):
    """Devolve (titulo, janelas). Uma janela por bloco '--- Janela #n ---'."""
    with open(caminho, encoding="utf-8") as f:
        linhas = f.readlines()

    titulo = os.path.basename(caminho)
    for linha in linhas:
        if linha.startswith("# Ensaio"):
            titulo = linha.lstrip("# ").strip()
            break

    janelas, atual = [], None
    for linha in linhas:
        m = _PADROES["janela"].match(linha)
        if m:
            if atual:
                janelas.append(atual)
            atual = {"indice": int(m.group(1)), "eixos": {}}
            continue
        if atual is None:
            continue

        if (m := _PADROES["amostra"].search(linha)):
            atual.update(fs_real=float(m.group(1)), duracao_ms=float(m.group(2)),
                         jitter_us=int(m.group(3)), atrasos=int(m.group(4)))
        elif (m := _PADROES["sensor"].search(linha)):
            atual.update(repetidas=int(m.group(1)), amostras=int(m.group(2)),
                         sensor_saturadas=int(m.group(3)),
                         erros_i2c=int(m.group(4)))
        elif (m := _PADROES["eixo"].search(linha)):
            atual["eixos"][m.group(1)] = {
                "media": float(m.group(2)), "dp": float(m.group(3)),
                "pico_min": float(m.group(4)), "pico_max": float(m.group(5))}
        elif (m := _PADROES["entrada"].search(linha)):
            atual.update(eixo=m.group(1), dc_removido=m.group(2) == "sim",
                         dp=float(m.group(3)), dp_treino=float(m.group(4)),
                         razao=float(m.group(5)))
        elif (m := _PADROES["quant"].search(linha)):
            atual.update(int8_min=int(m.group(1)), int8_max=int(m.group(2)),
                         niveis=int(m.group(3)), ceifadas=int(m.group(4)))
        elif (m := _PADROES["predicao"].match(linha)):
            atual.update(classe=m.group(1), confianca=float(m.group(2)) / 100.0,
                         inferencia_us=int(m.group(3)))

    if atual:
        janelas.append(atual)
    return titulo, janelas


def cabecalho(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def tabela(janelas):
    print(f"{'#':>5}{'dp (g)':>10}{'razao':>8}{'ceifadas':>10}{'%':>7}"
          f"{'niveis':>8}{'classe':>16}{'conf':>8}{'us':>8}")
    print("-" * 78)
    for j in janelas:
        frac = j["ceifadas"] / j["amostras"]
        marca = " <<<" if frac >= CEIFAMENTO_CRITICO else ""
        print(f"{j['indice']:>5}{j['dp']:>10.6f}{j['razao']:>7.2f}x"
              f"{j['ceifadas']:>10}{100 * frac:>6.1f}%{j['niveis']:>8}"
              f"{j['classe']:>16}{j['confianca']:>8.4f}"
              f"{j['inferencia_us']:>8}{marca}")


def aquisicao(janelas):
    cabecalho("AQUISICAO (ensaio V1 do protocolo, sob vibracao real)")

    fs = np.array([j["fs_real"] for j in janelas])
    jitter = np.array([j["jitter_us"] for j in janelas])
    repetidas = sum(j["repetidas"] for j in janelas)
    erros = sum(j["erros_i2c"] for j in janelas)
    saturadas = sum(j["sensor_saturadas"] for j in janelas)
    atrasos = sum(j["atrasos"] for j in janelas)
    total = sum(j["amostras"] for j in janelas)

    intervalo_us = 1e6 / fs.mean()
    linhas = [
        ("fs real", f"{fs.min():.1f} a {fs.max():.1f} Hz",
         abs(fs.mean() - 1000.0) / 1000.0 < 0.01),
        ("jitter maximo", f"{jitter.max()} us "
                          f"({100 * jitter.max() / intervalo_us:.1f}% do intervalo)",
         jitter.max() < 0.05 * intervalo_us),
        ("amostras repetidas", f"{repetidas} de {total}", repetidas == 0),
        ("saturacoes do sensor", f"{saturadas}", saturadas == 0),
        ("erros de I2C", f"{erros}", erros == 0),
        ("atrasos de laco", f"{atrasos}", atrasos == 0),
    ]
    for nome, valor, ok in linhas:
        print(f"  [{'ok   ' if ok else 'AVISO'}] {nome:<24} {valor}")


def calibracao(janelas):
    cabecalho("CALIBRACAO DO ACELEROMETRO")

    modulos = []
    for j in janelas:
        e = j["eixos"]
        if not all(k in e for k in "XYZ"):
            continue
        modulos.append(np.sqrt(sum(e[k]["media"] ** 2 for k in "XYZ")))
    if not modulos:
        return None

    modulos = np.array(modulos)
    desvio = 100 * (modulos.mean() - 1.0)
    print(f"  Modulo do vetor de gravidade: {modulos.mean():.4f} g "
          f"(min {modulos.min():.4f}, max {modulos.max():.4f})")
    print(f"  Esperado em repouso:          1,0000 g")
    print(f"  Desvio:                       {desvio:+.1f}%")
    print()
    if abs(desvio) > 5.0:
        print("  O modulo do vetor de gravidade nao depende da orientacao: uma")
        print("  inclinacao redistribui a gravidade entre os eixos, mas preserva")
        print("  o modulo. Um desvio deste tamanho e erro de calibracao, nao de")
        print("  montagem, e tem duas causas possiveis:")
        print()
        print("    - erro de SENSIBILIDADE (fator de escala): inflaria tambem")
        print("      todos os desvios padrao em g, e portanto as razoes de")
        print("      amplitude reportadas abaixo;")
        print("    - erro de OFFSET de zero-g no eixo Z: nao afeta desvio padrao")
        print("      nenhum, porque a constante some na subtracao da media.")
        print()
        # Com offset b: em repouso le 1+b; virada, le -1+b = (1+b)-2.
        # Com ganho k: em repouso le k; virada, le -k.
        print("  Para distinguir: vire a placa 180 graus e releia Z em repouso.")
        print(f"    offset        -> Z leria cerca de {modulos.mean() - 2.0:+.3f} g")
        print(f"    sensibilidade -> Z leria cerca de {-modulos.mean():+.3f} g")
    else:
        print("  Dentro do esperado para o MPU6050.")
    return modulos.mean()


def amplitude_e_ceifamento(janelas):
    cabecalho("AMPLITUDE E CEIFAMENTO")

    razoes = np.array([j["razao"] for j in janelas])
    fracoes = np.array([j["ceifadas"] / j["amostras"] for j in janelas])

    dentro = np.sum((razoes >= FAIXA_CONFIAVEL[0]) & (razoes <= FAIXA_CONFIAVEL[1]))
    criticas = np.sum(fracoes >= CEIFAMENTO_CRITICO)

    print(f"  Razao de amplitude: {razoes.min():.2f}x a {razoes.max():.2f}x "
          f"(mediana {np.median(razoes):.2f}x)")
    print(f"  Faixa em que o quantizador foi calibrado: "
          f"{FAIXA_CONFIAVEL[0]:.1f}x a {FAIXA_CONFIAVEL[1]:.1f}x")
    print(f"  Janelas dentro da faixa: {dentro} de {len(janelas)} "
          f"({100 * dentro / len(janelas):.0f}%)")
    print()
    print(f"  Ceifamento: {100 * fracoes.min():.1f}% a {100 * fracoes.max():.1f}%")
    print(f"  Janelas com ceifamento >= {100 * CEIFAMENTO_CRITICO:.0f}%: "
          f"{criticas} de {len(janelas)}")
    print()
    print("  Nenhuma predicao feita fora da faixa calibrada, ou com ceifamento")
    print("  acima do limite, deve ser citada como resultado de classificacao.")


def saida_do_modelo(janelas):
    cabecalho("SAIDA DO MODELO")

    confiancas = np.array([j["confianca"] for j in janelas])
    classes = [j["classe"] for j in janelas]

    print(f"  Confianca: {confiancas.min():.4f} a {confiancas.max():.4f} "
          f"(mediana {np.median(confiancas):.4f})")
    saturadas = np.sum(confiancas >= 0.99)
    print(f"  Janelas com confianca >= 99%: {saturadas} de {len(janelas)} "
          f"({100 * saturadas / len(janelas):.0f}%)")
    print()
    for classe in sorted(set(classes)):
        n = classes.count(classe)
        print(f"    {classe:<16} {n:>3} janelas ({100 * n / len(classes):>5.1f}%)")

    # A janela mais quieta do ensaio e o teste mais duro que existe: se o
    # sensor esta praticamente parado e o modelo ainda responde com confianca
    # maxima, ele nao tem estado de "nao sei".
    mais_quieta = min(janelas, key=lambda j: j["razao"])
    if mais_quieta["razao"] < FAIXA_CONFIAVEL[0]:
        print()
        print(f"  Janela mais quieta: #{mais_quieta['indice']}, "
              f"razao {mais_quieta['razao']:.2f}x "
              f"({mais_quieta['niveis']} niveis int8 distintos de 256).")
        print(f"    Predicao: {mais_quieta['classe']} com "
              f"{100 * mais_quieta['confianca']:.1f}% de confianca.")
        print(f"    Com o sensor praticamente em repouso, o modelo responde com")
        print(f"    confianca maxima. Nao ha estado de abstencao.")

    # Trocas de classe: o que mudou entre uma e outra?
    trocas = [(a, b) for a, b in zip(janelas, janelas[1:])
              if a["classe"] != b["classe"]]
    if trocas:
        print()
        print("  Trocas de classe:")
        for a, b in trocas:
            print(f"    #{a['indice']} -> #{b['indice']}: "
                  f"{a['classe']} -> {b['classe']}")
            print(f"      razao {a['razao']:.2f}x -> {b['razao']:.2f}x | "
                  f"ceifamento {100 * a['ceifadas'] / a['amostras']:.0f}% -> "
                  f"{100 * b['ceifadas'] / b['amostras']:.0f}%")


def modulacao(janelas):
    """A amplitude varia de forma periodica ao longo das janelas?

    Se a fonte de excitacao e pulsada, a envoltoria aparece como oscilacao do
    desvio padrao entre janelas. Detectar isso prova que a cadeia de aquisicao
    captura modulacao de amplitude - independentemente do que o modelo faca com
    ela.
    """
    if len(janelas) < 8:
        return
    cabecalho("MODULACAO DE AMPLITUDE ENTRE JANELAS")

    dp = np.array([j["dp"] for j in janelas])
    centrado = dp - dp.mean()
    if np.allclose(centrado, 0):
        return

    # Autocorrelacao normalizada; o primeiro pico depois do atraso zero da o
    # periodo da envoltoria, em numero de janelas.
    auto = np.correlate(centrado, centrado, mode="full")[len(centrado) - 1:]
    auto = auto / auto[0]

    periodo = None
    for k in range(2, len(auto) - 1):
        if auto[k] > auto[k - 1] and auto[k] >= auto[k + 1] and auto[k] > 0.2:
            periodo = k
            break

    print(f"  Desvio padrao do eixo {janelas[0]['eixo']}: "
          f"{dp.min():.4f} a {dp.max():.4f} g "
          f"(variacao de {100 * (dp.max() / dp.min() - 1):.0f}%)")
    print(f"  Autocorrelacao: " +
          "  ".join(f"k={k}:{auto[k]:+.2f}" for k in range(1, min(7, len(auto)))))
    if periodo:
        duracao = np.mean([j["duracao_ms"] for j in janelas]) / 1000.0
        print()
        print(f"  Periodicidade detectada: {periodo} janelas "
              f"(correlacao {auto[periodo]:+.2f}).")
        print(f"  A envoltoria da excitacao foi capturada pela aquisicao.")
        print(f"  Em tempo, isso equivale a {periodo} x o intervalo entre")
        print(f"  janelas; se elas fossem contiguas ({duracao:.3f} s cada), o")
        print(f"  periodo seria {periodo * duracao:.2f} s.")
        print()
        print("  A saida do classificador NAO acompanha essa modulacao - ver a")
        print("  secao anterior. O sinal esta la; o modelo e cego a ele.")
    else:
        print("  Nenhuma periodicidade clara na envoltoria.")


def figura(titulo, janelas, caminho):
    indices = [j["indice"] for j in janelas]
    razoes = [j["razao"] for j in janelas]
    fracoes = [100 * j["ceifadas"] / j["amostras"] for j in janelas]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(indices, razoes, "o-", color="#37474F", lw=1.5)
    axes[0].axhspan(*FAIXA_CONFIAVEL, color="#2E7D32", alpha=0.12)
    axes[0].axhline(FAIXA_CONFIAVEL[1], color="#2E7D32", ls="--", lw=1)
    axes[0].annotate("faixa calibrada do quantizador", xy=(indices[0], 1.25),
                     fontsize=9, color="#2E7D32")
    axes[0].set_ylabel("amplitude / treino")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(titulo, fontsize=12)

    axes[1].bar(indices, fracoes, color="#EF6C00", width=0.6)
    axes[1].axhline(100 * CEIFAMENTO_CRITICO, color="#B71C1C", ls="--", lw=1.2)
    axes[1].annotate(f"{100 * CEIFAMENTO_CRITICO:.0f}% — janela deformada",
                     xy=(indices[0], 100 * CEIFAMENTO_CRITICO + 0.6),
                     fontsize=9, color="#B71C1C")
    axes[1].set_ylabel("amostras ceifadas [%]")
    axes[1].grid(alpha=0.3, axis="y")

    for j in janelas:
        axes[2].bar(j["indice"], j["confianca"], width=0.6,
                    color=CORES_CLASSE.get(j["classe"], "#616161"))
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("confianca")
    axes[2].set_xlabel("janela")
    axes[2].grid(alpha=0.3, axis="y")
    vistas = sorted({j["classe"] for j in janelas})
    axes[2].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=CORES_CLASSE.get(c, "#616161"), label=c)
        for c in vistas], fontsize=9, loc="lower right")

    fig.suptitle("Sem rolamento sob o sensor: a classe predita nao tem "
                 "significado fisico", fontsize=10, y=0.995, color="#B71C1C")
    plt.tight_layout()
    plt.savefig(caminho, dpi=200)
    plt.close()
    return caminho


def analisar(caminho):
    titulo, janelas = ler_captura(caminho)
    if not janelas:
        print(f"[AVISO] Nenhuma janela reconhecida em {caminho}.")
        return None

    print()
    print("#" * 78)
    print(f"# {titulo}")
    print(f"# {os.path.basename(caminho)} | {len(janelas)} janelas "
          f"(#{janelas[0]['indice']} a #{janelas[-1]['indice']})")
    print("#" * 78)

    cabecalho("JANELA A JANELA")
    tabela(janelas)
    aquisicao(janelas)
    calibracao(janelas)
    amplitude_e_ceifamento(janelas)
    saida_do_modelo(janelas)
    modulacao(janelas)

    destino = os.path.splitext(caminho)[0] + ".png"
    figura(titulo, janelas, destino)
    print()
    print(f"Figura: {destino}")
    return janelas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capturas", nargs="+",
                        help="arquivos .txt com a captura do console serial")
    args = parser.parse_args()

    caminhos = []
    for padrao in args.capturas:
        caminhos.extend(sorted(glob.glob(padrao)) or [padrao])

    for caminho in caminhos:
        if not os.path.exists(caminho):
            print(f"[AVISO] {caminho} nao existe.")
            continue
        analisar(caminho)

    print()
    print("=" * 78)
    print("Lembrete: nenhum destes ensaios valida CLASSIFICACAO de falha. Sem")
    print("rolamento girando sob carga nao existem BPFI/BPFO, e qualquer classe")
    print("predita e ruido. O que eles validam e a cadeia de aquisicao e o")
    print("comportamento do sistema fora da distribuicao de treino.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
