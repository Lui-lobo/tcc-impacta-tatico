"""Etapa 2 - Diagnostico do dataset bruto (o "ANTES").

Mede o estado do dataset SEM nenhuma correcao aplicada: le como o
src/pipeline/data_processor.py le hoje, com a selecao ingenua de canal, e
registra tudo que estiver errado.

O resultado e gravado em relatorios/diagnostico_antes.json. A etapa 4 recalcula
exatamente as mesmas medidas sobre o dataset curado e compara. Por isso este
modulo nao corrige nada: se corrigisse, nao haveria "antes".

    python -m dataset_engineering.etapa2_diagnostico
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset_engineering import config, leitura, metricas

SAIDA = os.path.join(config.DIR_RELATORIOS, "diagnostico_antes.json")


def cabecalho(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def carregar_como_o_pipeline_le():
    """Reproduz a leitura do build.py: primeira chave '_DE_time' que aparecer.

    Nao chama o DataProcessor diretamente para conseguir registrar, arquivo a
    arquivo, qual chave foi escolhida - informacao que o pipeline descarta.
    """
    registros = []
    for ensaio in config.ENSAIOS:
        if not os.path.exists(ensaio.caminho):
            continue
        bruto, diagnostico = leitura.ler_bruto(ensaio, estrito=False)
        # A leitura ingenua: pega a primeira chave, seja ela qual for.
        import scipy.io
        mat = scipy.io.loadmat(ensaio.caminho)
        ingenua = diagnostico["chave_ingenua"]
        serie = mat[ingenua].flatten().astype(np.float64) if ingenua else bruto
        registros.append({
            "ensaio": ensaio,
            "bruto": serie,
            "dec": leitura.decimar(serie, ensaio.taxa_origem),
            "rpm_medido": leitura.rpm_medido(ensaio),
            **diagnostico,
            "chave_usada": ingenua,
            # A leitura ingenua devolve a serie errada exatamente nos arquivos
            # em que a primeira chave nao e a do proprio ensaio.
            "leitura_incorreta": diagnostico["divergente"],
        })
    return registros


def integridade(registros):
    cabecalho("2.1 INTEGRIDADE - cada arquivo entrega a propria serie?")

    divergentes = [r for r in registros if r["divergente"]]
    print(f"{'arquivo':<12}{'chave lida':<16}{'chave correta':<16}"
          f"{'amostras':>10}  situacao")
    print("-" * 78)
    for r in registros:
        e = r["ensaio"]
        marca = "<<< SERIE DE OUTRO ENSAIO" if r["divergente"] else "ok"
        print(f"{e.arquivo:<12}{str(r['chave_usada']):<16}{e.chave_mat:<16}"
              f"{len(r['bruto']):>10}  {marca}")

    # Conteudo identico entre arquivos diferentes.
    assinaturas = {}
    for r in registros:
        assinaturas.setdefault(hash(r["bruto"].tobytes()), []).append(
            r["ensaio"].arquivo)
    duplicados = [v for v in assinaturas.values() if len(v) > 1]

    print("-" * 78)
    print(f"Arquivos lendo a serie errada: {len(divergentes)}")
    print(f"Grupos com conteudo identico:  {len(duplicados)}")
    for grupo in duplicados:
        print(f"   {' == '.join(grupo)}")
    return divergentes, duplicados


def observabilidade(diagnostico):
    cabecalho("2.2 OBSERVABILIDADE - a assinatura do defeito existe no sinal?")

    print(f"Razao pico/mediana do envelope na frequencia caracteristica.")
    print(f"Piso de ruido adotado: {config.PISO_ENVELOPE:.0f}.")
    print()
    print(f"{'arquivo':<12}{'classe':<15}{'sever.':<9}{'carga':>6}"
          f"{'BPFx':>8}{'2k-5kHz':>11}{'apos 1kHz':>11}   situacao")
    print("-" * 78)

    mudos = []
    for m in diagnostico["medidas"]:
        if m["classe"] == 0:
            continue
        res, dec = m["envelope_ressonancia"], m["envelope_decimado"]
        if not m["observavel"]:
            situacao = "SEM assinatura nem no original"
            mudos.append(m)
        elif dec < config.PISO_ENVELOPE:
            situacao = "perdida na decimacao"
            mudos.append(m)
        else:
            situacao = f"sobrevive ({res / dec:.0f}x mais fraca)" if res > dec \
                else "sobrevive"
        print(f"{m['numero']:<12}{config.NOMES_CURTOS[m['classe']]:<15}"
              f"{(m['severidade'] or '-'):<9}{m['carga']:>4} hp"
              f"{m['frequencia_defeito']:>8.1f}{res:>11.1f}{dec:>11.1f}"
              f"   {situacao}")

    print("-" * 78)
    print(f"Ensaios de falha sem assinatura utilizavel: {len(mudos)}")
    for m in mudos:
        print(f"   {m['numero']}.mat ({config.NOMES_CURTOS[m['classe']]}, "
              f"{m['severidade']}, {m['carga']} hp)")
    return mudos


def perda_espectral(diagnostico):
    cabecalho("2.3 PERDA ESPECTRAL - o que a decimacao descarta")

    print(f"{'classe':<16}{'0-500':>9}{'500-2k':>9}{'2k-5k':>9}{'>5k':>9}"
          f"{'descartado':>13}")
    print("-" * 78)
    for classe, pasta in config.CLASSES.items():
        grupo = [m for m in diagnostico["medidas"] if m["classe"] == classe]
        if not grupo:
            continue
        chaves = [f"0-{config.NYQUIST_ALVO_HZ}",
                  f"{config.NYQUIST_ALVO_HZ}-2k", "2k-5k", ">5k", "descartado"]
        medias = [100 * np.mean([m["origem"].get(k, 0.0) for m in grupo])
                  for k in chaves]
        print(f"{pasta:<16}" + "".join(f"{v:>8.1f}%" for v in medias[:4]) +
              f"{medias[4]:>12.1f}%")

    print()
    print(f"{'classe':<16}{'curtose orig':>14}{'curtose 1kHz':>14}"
          f"{'RMS 1kHz min':>15}{'RMS 1kHz max':>15}")
    print("-" * 78)
    for classe, pasta in config.CLASSES.items():
        grupo = [m for m in diagnostico["medidas"] if m["classe"] == classe]
        if not grupo:
            continue
        rms = [m["decimado"]["rms"] for m in grupo]
        print(f"{pasta:<16}"
              f"{np.mean([m['origem']['curtose'] for m in grupo]):>14.2f}"
              f"{np.mean([m['decimado']['curtose'] for m in grupo]):>14.2f}"
              f"{min(rms):>15.4f}{max(rms):>15.4f}")

    sobrepostas = diagnostico["rms"]["sobrepostas"]
    if sobrepostas:
        print()
        print(f"A faixa de RMS da classe normal a {config.TAXA_ALVO_HZ} Hz se "
              f"sobrepoe a de: {', '.join(sobrepostas)}.")
        print("   Nenhum limiar de amplitude separa as classes na taxa do sensor.")


def escala_e_balanco(diagnostico):
    cabecalho("2.4 ESCALA E BALANCO")

    esc = diagnostico["escala"]
    print(f"{esc['arquivos']} arquivos | {esc['duracao_s']:.1f} s a "
          f"{config.TAXA_ALVO_HZ} Hz | {esc['janelas_independentes']} janelas "
          f"independentes")
    print()
    print(f"{'classe':<16}{'arquivos':>10}{'segundos':>11}{'% total':>10}"
          f"{'janelas indep.':>16}")
    print("-" * 78)
    for pasta, v in esc["por_classe"].items():
        print(f"{pasta:<16}{v['arquivos']:>10}{v['duracao_s']:>11.1f}"
              f"{100 * v['fracao']:>9.1f}%{v['janelas_independentes']:>16}")
    print("-" * 78)
    print(f"Desbalanceamento (maior/menor classe): "
          f"{esc['desbalanceamento']:.1f} : 1")


def sonda(diagnostico):
    cabecalho("2.5 SONDA DE SEPARABILIDADE (regressao logistica, "
              "deixando uma carga de fora)")

    print("Classificador LINEAR sobre atributos rasos, so para medir se a")
    print("informacao esta acessivel no dado. Nao e o modelo do projeto.")
    print()
    s = diagnostico["sonda"]
    if not s["rodadas"]:
        print("Sem rodadas validas.")
        return

    print(f"{'carga':>7}{'rpm':>7}{'jan. teste':>12}{'acuracia':>11}"
          f"{'deteccao':>11}{'f. alarme':>12}")
    print("-" * 78)
    for r in s["rodadas"]:
        det = "n/d" if np.isnan(r["deteccao"]) else f"{r['deteccao'] * 100:.2f}%"
        fa = "n/d" if np.isnan(r["falso_alarme"]) else f"{r['falso_alarme'] * 100:.2f}%"
        print(f"{r['carga']:>4} hp{config.RPM_NOMINAL[r['carga']]:>7}"
              f"{r['n_teste']:>12}{r['acuracia'] * 100:>10.2f}%"
              f"{det:>11}{fa:>12}")
    print("-" * 78)
    print(f"Acuracia media:  {s['acuracia_media'] * 100:.2f}% "
          f"(desvio {s['acuracia_desvio'] * 100:.2f} pp)")
    print(f"Deteccao media:  {s['deteccao_media'] * 100:.2f}%")
    print(f"Falso alarme:    {s['falso_alarme_medio'] * 100:.2f}%")


def _serializavel(obj):
    if isinstance(obj, dict):
        return {k: _serializavel(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializavel(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def executar(verbose=True):
    print("ETAPA 2 - Diagnostico do dataset bruto (o ANTES)")
    print("Leitura reproduzindo o comportamento atual do build.py.")

    registros = carregar_como_o_pipeline_le()
    if not registros:
        print("\nNenhum .mat encontrado. Execute 'python download_cwru.py'.")
        return None

    divergentes, duplicados = integridade(registros)
    diagnostico = metricas.diagnostico_completo(registros)

    observabilidade(diagnostico)
    perda_espectral(diagnostico)
    escala_e_balanco(diagnostico)
    sonda(diagnostico)

    ausentes = [e.arquivo for e in config.ENSAIOS
                if not os.path.exists(e.caminho)]
    diagnostico["ausentes"] = ausentes
    diagnostico["duplicados"] = duplicados
    diagnostico["rotulo"] = "antes"

    config.garantir_diretorios()
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(_serializavel(diagnostico), f, indent=2, ensure_ascii=False)

    cabecalho("2.6 RESUMO DO DIAGNOSTICO")
    achados = [
        (len(ausentes), "arquivo(s) do catalogo ausentes"),
        (len(divergentes), "arquivo(s) lendo a serie de outro ensaio"),
        (len(duplicados), "grupo(s) de arquivos com conteudo identico"),
        (len(diagnostico["nao_observaveis"]),
         "ensaio(s) de falha sem assinatura observavel"),
    ]
    for quantidade, descricao in achados:
        print(f"  [{'ok   ' if quantidade == 0 else 'AVISO'}] "
              f"{quantidade:>2}  {descricao}")
    print()
    print(f"Gravado em {os.path.relpath(SAIDA, config.RAIZ)}")
    return diagnostico


if __name__ == "__main__":
    executar()
