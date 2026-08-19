"""Etapa 4 - Comparacao ANTES x DEPOIS e ablacao por tratativa.

Recalcula sobre o dataset curado exatamente as mesmas medidas que a etapa 2
calculou sobre o bruto - mesmo codigo, mesmo protocolo - e confronta os dois.

ABLACAO
-------
Comparar apenas "antes" com "depois" diria que o conjunto melhorou, mas nao
qual tratativa produziu a melhora. Por isso a etapa mede tres conjuntos:

    bruto        como o build.py le hoje (canal ingenuo, 28 ensaios)
    +T1          canal corrigido, ainda com os 28 ensaios
    +T1+T9       canal corrigido e ensaios sem assinatura em quarentena

A diferenca entre o primeiro e o segundo isola o efeito da correcao de leitura.
A diferenca entre o segundo e o terceiro isola o efeito da quarentena - que
custa dados, e portanto precisa justificar o proprio custo.

    python -m dataset_engineering.etapa4_comparacao
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset_engineering import config, leitura, metricas
from dataset_engineering.etapa2_diagnostico import carregar_como_o_pipeline_le

SAIDA = os.path.join(config.DIR_RELATORIOS, "comparacao.json")


def cabecalho(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def carregar_curado():
    """Le data_curado/ a partir do manifesto, ja decimado."""
    if not os.path.exists(config.MANIFESTO):
        raise FileNotFoundError(
            f"{config.MANIFESTO} nao existe. Rode a etapa 3 primeiro.")

    with open(config.MANIFESTO, encoding="utf-8") as f:
        manifesto = json.load(f)

    registros = []
    for entrada in manifesto["ensaios"]:
        if entrada["status"] != "incluido":
            continue
        ensaio = config.POR_NUMERO[entrada["numero"]]
        serie = np.load(os.path.join(config.DIR_CURADO, entrada["npy"]))
        registros.append({
            "ensaio": ensaio,
            # A serie curada JA esta a 1 kHz. Guardar a mesma nos dois campos
            # mantem a interface das metricas, mas as medidas de "origem"
            # passam a descrever o sinal decimado - o que e correto: e este o
            # sinal que o dataset curado entrega.
            "bruto": serie.astype(np.float64),
            "dec": serie.astype(np.float64),
            "rpm_medido": entrada["rpm_medido"],
            "chave_usada": entrada["chave_mat"],
            "chave_ingenua": entrada["chave_ingenua"],
            "divergente": entrada["corrigido_t1"],
            "leitura_incorreta": False,
        })
    return manifesto, registros


def conjunto_t1():
    """Canal corrigido, sem quarentena. O passo intermediario da ablacao."""
    registros, _ = leitura.carregar_ensaios(estrito=True)
    return registros


def _sonda(registros):
    return metricas.sonda_linear(registros)


def _escala(registros):
    return metricas.escala(metricas.medir_conjunto(registros))


def comparar_escala(conjuntos):
    cabecalho("4.1 ESCALA - o que entra no treino")

    print(f"{'conjunto':<16}{'ensaios':>9}{'segundos':>11}"
          f"{'jan. indep.':>13}{'desbalanc.':>13}")
    print("-" * 78)
    resultado = {}
    for nome, registros in conjuntos.items():
        esc = _escala(registros)
        resultado[nome] = esc
        print(f"{nome:<16}{esc['arquivos']:>9}{esc['duracao_s']:>11.1f}"
              f"{esc['janelas_independentes']:>13}"
              f"{esc['desbalanceamento']:>11.1f}:1")

    print()
    print(f"{'conjunto':<16}" +
          "".join(f"{p.split('_', 1)[1][:12]:>16}" for p in config.CLASSES.values()))
    print("-" * 78)
    for nome, esc in resultado.items():
        celulas = []
        for pasta in config.CLASSES.values():
            v = esc["por_classe"][pasta]
            celulas.append(f"{v['janelas_independentes']:>6} ({100 * v['fracao']:>4.1f}%)")
        print(f"{nome:<16}" + "".join(c.rjust(16) for c in celulas))
    return resultado


def comparar_sonda(conjuntos):
    cabecalho("4.2 SEPARABILIDADE - sonda linear, deixando uma carga de fora")

    print("Mesmo protocolo do tools/validacao_por_carga.py: treina em tres")
    print("cargas, testa na quarta, janelas independentes dos dois lados.")
    print()
    print(f"{'conjunto':<16}{'acuracia':>11}{'desvio':>10}"
          f"{'deteccao':>11}{'f. alarme':>12}")
    print("-" * 78)

    resultado = {}
    for nome, registros in conjuntos.items():
        s = _sonda(registros)
        resultado[nome] = s
        print(f"{nome:<16}{s['acuracia_media'] * 100:>10.2f}%"
              f"{s['acuracia_desvio'] * 100:>9.2f}"
              f"{s['deteccao_media'] * 100:>10.2f}%"
              f"{s['falso_alarme_medio'] * 100:>11.2f}%")

    print()
    print("Por carga de teste:")
    print(f"{'conjunto':<16}" + "".join(f"{c} hp".rjust(15) for c in config.CARGAS))
    print("-" * 78)
    for nome, s in resultado.items():
        por_carga = {r["carga"]: r["acuracia"] for r in s["rodadas"]}
        print(f"{nome:<16}" + "".join(
            (f"{100 * por_carga[c]:>13.2f}%" if c in por_carga else f"{'n/d':>14}")
            for c in config.CARGAS))

    print()
    print("Deteccao de falha por carga (fracao das janelas com falha que NAO")
    print("foram chamadas de normal):")
    print(f"{'conjunto':<16}" + "".join(f"{c} hp".rjust(15) for c in config.CARGAS))
    print("-" * 78)
    for nome, s in resultado.items():
        por_carga = {r["carga"]: r["deteccao"] for r in s["rodadas"]}
        celulas = []
        for c in config.CARGAS:
            v = por_carga.get(c)
            celulas.append(f"{'n/d':>14}" if v is None or np.isnan(v)
                           else f"{100 * v:>13.2f}%")
        print(f"{nome:<16}" + "".join(celulas))
    return resultado


def comparar_integridade(conjuntos):
    cabecalho("4.3 INTEGRIDADE E OBSERVABILIDADE")

    print(f"{'conjunto':<16}{'canal errado':>15}{'duplicados':>13}"
          f"{'sem assinatura':>17}")
    print("-" * 78)
    for nome, registros in conjuntos.items():
        divergentes = sum(1 for r in registros if r.get("leitura_incorreta"))
        assinaturas = {}
        for r in registros:
            assinaturas.setdefault(hash(r["bruto"].tobytes()), 0)
            assinaturas[hash(r["bruto"].tobytes())] += 1
        duplicados = sum(1 for v in assinaturas.values() if v > 1)

        mudos = 0
        for r in registros:
            e = r["ensaio"]
            if e.classe == 0:
                continue
            # No conjunto curado o sinal ja esta a 1 kHz e a banda de
            # ressonancia nao existe mais; a medida usa a serie inteira.
            mesma = r["bruto"] is r["dec"] or len(r["bruto"]) == len(r["dec"])
            banda = None if mesma else config.BANDA_RESSONANCIA_HZ
            fs = config.TAXA_ALVO_HZ if mesma else e.taxa_origem
            razao = metricas.razao_envelope(
                r["bruto"], fs, e.frequencia_defeito, banda=banda)
            if not (razao >= config.PISO_ENVELOPE):
                mudos += 1

        print(f"{nome:<16}{divergentes:>15}{duplicados:>13}{mudos:>17}")


def relatorio(escalas, sondas, manifesto):
    """Grava o comparativo em JSON, para o relatorio final montar as tabelas."""
    config.garantir_diretorios()
    pacote = {
        "escalas": escalas,
        "sondas": {nome: {k: v for k, v in s.items() if k != "rodadas"}
                   | {"rodadas": s["rodadas"]}
                   for nome, s in sondas.items()},
        "manifesto": {k: manifesto[k] for k in ("gerado_em", "totais",
                                                "tratativas")},
    }
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(pacote, f, indent=2, ensure_ascii=False, default=float)
    return pacote


def executar():
    print("ETAPA 4 - Comparacao ANTES x DEPOIS")

    manifesto, curado = carregar_curado()

    conjuntos = {
        "bruto": carregar_como_o_pipeline_le(),
        "+T1": conjunto_t1(),
        "+T1+T9": curado,
    }

    escalas = comparar_escala(conjuntos)
    sondas = comparar_sonda(conjuntos)
    comparar_integridade(conjuntos)

    cabecalho("4.4 VEREDITO")

    base = sondas["bruto"]
    final = sondas["+T1+T9"]
    delta_acc = (final["acuracia_media"] - base["acuracia_media"]) * 100
    delta_det = (final["deteccao_media"] - base["deteccao_media"]) * 100
    delta_fa = (final["falso_alarme_medio"] - base["falso_alarme_medio"]) * 100

    print(f"{'medida':<28}{'antes':>10}{'depois':>10}{'delta':>12}")
    print("-" * 78)
    for nome, antes, depois, unidade in (
        ("Acuracia (sonda linear)", base["acuracia_media"] * 100,
         final["acuracia_media"] * 100, "pp"),
        ("Deteccao de falha", base["deteccao_media"] * 100,
         final["deteccao_media"] * 100, "pp"),
        ("Falso alarme", base["falso_alarme_medio"] * 100,
         final["falso_alarme_medio"] * 100, "pp"),
        ("Desvio entre cargas", base["acuracia_desvio"] * 100,
         final["acuracia_desvio"] * 100, "pp"),
    ):
        print(f"{nome:<28}{antes:>9.2f}%{depois:>9.2f}%"
              f"{depois - antes:>+10.2f} {unidade}")

    ea, ef = escalas["bruto"], escalas["+T1+T9"]
    print(f"{'Ensaios':<28}{ea['arquivos']:>10}{ef['arquivos']:>10}"
          f"{ef['arquivos'] - ea['arquivos']:>+12}")
    print(f"{'Janelas independentes':<28}{ea['janelas_independentes']:>10}"
          f"{ef['janelas_independentes']:>10}"
          f"{ef['janelas_independentes'] - ea['janelas_independentes']:>+12}")

    print()
    if delta_acc > 0:
        print(f"A curadoria melhorou a separabilidade em {delta_acc:+.2f} pp "
              f"com {ea['arquivos'] - ef['arquivos']} ensaios a MENOS.")
    else:
        print(f"A curadoria nao melhorou a sonda ({delta_acc:+.2f} pp). "
              f"Ela corrigiu a CORRECAO dos rotulos, o que vale por si.")
    print("A sonda e linear e rasa: serve para comparar conjuntos, nao para")
    print("prever o desempenho da CNN. Rode o build.py para o numero real.")

    pacote = relatorio(escalas, sondas, manifesto)
    print()
    print(f"Gravado em {os.path.relpath(SAIDA, config.RAIZ)}")
    return pacote


if __name__ == "__main__":
    executar()
