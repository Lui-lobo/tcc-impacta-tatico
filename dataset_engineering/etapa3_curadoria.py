"""Etapa 3 - Curadoria: aplica as tratativas e materializa data_curado/.

TRATATIVAS APLICADAS
--------------------
T1  Selecao de canal pelo NUMERO do arquivo, e nao pela ordem das chaves do
    .mat. Corrige 99.mat, que carrega tambem as variaveis de 98.mat e por isso
    entregava a serie de 1 hp rotulada como 2 hp.

T2  Inventario obrigatorio. Se faltar qualquer arquivo do catalogo, a curadoria
    para. Treinar em silencio com um dataset incompleto foi exatamente o que
    produziu os numeros dessincronizados dos relatorios anteriores.

T3  Deduplicacao. Series identicas amostra a amostra sao detectadas e a
    repetida vai para quarentena. Depois de T1 nao deve sobrar nenhuma - a
    verificacao continua porque um dataset baixado de novo pode reintroduzir o
    problema.

T9  Quarentena por OBSERVABILIDADE. Um ensaio de falha cuja frequencia
    caracteristica nao se destaca do ruido - nem no sinal original, antes de
    qualquer decimacao - carrega um rotulo que o sinal nao sustenta. E um
    criterio medido, nao uma lista negra: se o CWRU publicar novos arquivos, o
    mesmo teste se aplica a eles.

O QUE E GRAVADO
---------------
data_curado/
    manifesto.json            metadados de todos os 28 ensaios, incluidos e em
                              quarentena, com o motivo de cada exclusao
    0_normal/097.npy          serie ja decimada para 1 kHz, float32
    1_inner_race/105.npy
    ...

Guardar a serie JA decimada e deliberado: a decimacao e deterministica e cara
(o filtro anti-aliasing roda em varios estagios), e o build.py, o
validacao_por_carga.py e este fluxo passam a partir do MESMO sinal, byte a
byte. Antes, cada ferramenta decimava por conta propria.

    python -m dataset_engineering.etapa3_curadoria
    python -m dataset_engineering.etapa3_curadoria --manter-suspeitos
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset_engineering import config, leitura, metricas


def cabecalho(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def verificar_inventario():
    """T2 - o catalogo esta completo?"""
    cabecalho("3.1 INVENTARIO (T2)")

    ausentes = [e for e in config.ENSAIOS if not os.path.exists(e.caminho)]
    print(f"Catalogados: {len(config.ENSAIOS)} | "
          f"Presentes: {len(config.ENSAIOS) - len(ausentes)} | "
          f"Ausentes: {len(ausentes)}")
    if ausentes:
        print()
        for e in ausentes:
            print(f"   AUSENTE {e.pasta}/{e.arquivo}  {e.rotulo}")
        print()
        print("   A curadoria NAO prossegue com o catalogo incompleto.")
        print("   Execute 'python download_cwru.py' e repita.")
    else:
        print("Inventario completo.")
    return ausentes


def aplicar_correcao_canal():
    """T1 - le todos os ensaios com a chave derivada do numero do arquivo."""
    cabecalho("3.2 CORRECAO DE CANAL (T1)")

    registros, _ = leitura.carregar_ensaios(estrito=True)
    corrigidos = [r for r in registros if r["divergente"]]

    if corrigidos:
        print(f"{len(corrigidos)} arquivo(s) corrigidos:")
        for r in corrigidos:
            e = r["ensaio"]
            print(f"   {e.arquivo:<10} lia '{r['chave_ingenua']}' -> "
                  f"agora le '{r['chave_usada']}'  "
                  f"({e.carga} hp, {len(r['bruto'])} amostras)")
    else:
        print("Nenhuma correcao necessaria: cada arquivo ja entregava a "
              "propria serie.")
    return registros, corrigidos


def detectar_duplicados(registros):
    """T3 - series identicas entre arquivos diferentes."""
    cabecalho("3.3 DEDUPLICACAO (T3)")

    vistos, repetidos = {}, []
    for r in registros:
        digest = hashlib.sha256(r["bruto"].tobytes()).hexdigest()
        r["sha256"] = digest
        if digest in vistos:
            repetidos.append((r, vistos[digest]))
        else:
            vistos[digest] = r

    if repetidos:
        print(f"{len(repetidos)} arquivo(s) duplicados:")
        for r, original in repetidos:
            print(f"   {r['ensaio'].arquivo} == {original['ensaio'].arquivo}  "
                  f"-> quarentena")
    else:
        print("Nenhum conteudo duplicado.")
    return {r["ensaio"].numero for r, _ in repetidos}


def avaliar_observabilidade(registros):
    """T9 - a assinatura do defeito existe no sinal original?"""
    cabecalho("3.4 OBSERVABILIDADE (T9)")

    print(f"Criterio: razao pico/mediana do espectro de envelope na frequencia")
    print(f"caracteristica, medida no sinal ORIGINAL demodulado em "
          f"{config.BANDA_RESSONANCIA_HZ[0] // 1000}-"
          f"{config.BANDA_RESSONANCIA_HZ[1] // 1000} kHz.")
    print(f"Piso de ruido: {config.PISO_ENVELOPE:.0f}.")
    print()

    razoes, reprovados = {}, set()
    for r in registros:
        e = r["ensaio"]
        if e.classe == 0:
            continue
        razao = metricas.razao_envelope(
            r["bruto"], e.taxa_origem, e.frequencia_defeito,
            banda=config.BANDA_RESSONANCIA_HZ)
        razoes[e.numero] = razao
        if not (razao >= config.PISO_ENVELOPE):
            reprovados.add(e.numero)

    aprovados = len(razoes) - len(reprovados)
    print(f"Ensaios de falha avaliados: {len(razoes)} | "
          f"aprovados: {aprovados} | reprovados: {len(reprovados)}")
    if reprovados:
        print()
        for numero in sorted(reprovados):
            e = config.POR_NUMERO[numero]
            print(f"   {e.arquivo:<10} razao={razoes[numero]:>7.1f}  "
                  f"({config.NOMES_CURTOS[e.classe]}, {e.severidade}, "
                  f"{e.carga} hp)")
        print()
        print("   O defeito existe na bancada, mas nao se manifesta no sinal.")
        print("   Mantidos, seriam rotulos que nenhum sensor poderia sustentar.")
    return razoes, reprovados


def gravar(registros, quarentena, motivos, razoes):
    """Materializa data_curado/ e o manifesto."""
    cabecalho("3.5 GRAVACAO")

    if os.path.isdir(config.DIR_CURADO):
        shutil.rmtree(config.DIR_CURADO)
    for pasta in config.CLASSES.values():
        os.makedirs(os.path.join(config.DIR_CURADO, pasta), exist_ok=True)

    entradas, incluidos = [], 0
    for r in registros:
        e = r["ensaio"]
        em_quarentena = e.numero in quarentena
        relativo = os.path.join(e.pasta, f"{e.numero:03d}.npy")

        entrada = {
            "numero": e.numero,
            "arquivo_origem": e.arquivo,
            "classe": e.classe,
            "pasta": e.pasta,
            "severidade": e.severidade,
            "carga_hp": e.carga,
            "rpm_nominal": e.rpm,
            "rpm_medido": r["rpm_medido"],
            "frequencia_defeito_hz": round(e.frequencia_defeito, 2),
            "fs_origem_hz": e.taxa_origem,
            "fs_hz": config.TAXA_ALVO_HZ,
            "chave_mat": r["chave_usada"],
            "chave_ingenua": r["chave_ingenua"],
            "corrigido_t1": r["divergente"],
            "amostras_origem": int(len(r["bruto"])),
            "amostras": int(len(r["dec"])),
            "duracao_s": round(len(r["dec"]) / config.TAXA_ALVO_HZ, 3),
            "envelope_ressonancia": (None if e.numero not in razoes
                                     else round(float(razoes[e.numero]), 1)),
            "sha256_origem": r["sha256"],
            "status": "quarentena" if em_quarentena else "incluido",
            "motivo": motivos.get(e.numero),
            "npy": None if em_quarentena else relativo.replace(os.sep, "/"),
        }
        entradas.append(entrada)

        if not em_quarentena:
            np.save(os.path.join(config.DIR_CURADO, relativo),
                    r["dec"].astype(np.float32))
            incluidos += 1

    manifesto = {
        "versao": 2,
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origem": os.path.relpath(config.DIR_BRUTO, config.RAIZ),
        "fs_hz": config.TAXA_ALVO_HZ,
        "tamanho_janela": config.TAMANHO_JANELA,
        "classes": {str(k): v for k, v in config.CLASSES.items()},
        "tratativas": {
            "T1_selecao_de_canal_por_numero": True,
            "T2_inventario_obrigatorio": True,
            "T3_deduplicacao": True,
            "T9_quarentena_por_observabilidade": {
                "aplicada": any(m == "sem assinatura observavel"
                                for m in motivos.values()),
                "piso_envelope": config.PISO_ENVELOPE,
                "banda_ressonancia_hz": list(config.BANDA_RESSONANCIA_HZ),
            },
        },
        "totais": {
            "catalogados": len(config.ENSAIOS),
            "incluidos": incluidos,
            "quarentena": len(quarentena),
        },
        "ensaios": entradas,
    }

    with open(config.MANIFESTO, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, indent=2, ensure_ascii=False)

    amostras = sum(e["amostras"] for e in entradas
                   if e["status"] == "incluido")
    print(f"Incluidos: {incluidos} | Quarentena: {len(quarentena)}")
    print(f"Sinal curado: {amostras} amostras a {config.TAXA_ALVO_HZ} Hz = "
          f"{amostras / config.TAXA_ALVO_HZ:.1f} s "
          f"({amostras // config.TAMANHO_JANELA} janelas independentes)")
    print()
    print(f"   {os.path.relpath(config.MANIFESTO, config.RAIZ)}")
    print(f"   {os.path.relpath(config.DIR_CURADO, config.RAIZ)}/<classe>/<numero>.npy")

    if quarentena:
        print()
        print("Em quarentena (gravados no manifesto, nao no disco):")
        for numero in sorted(quarentena):
            print(f"   {numero}.mat  -> {motivos[numero]}")
    return manifesto


def executar(manter_suspeitos=False):
    print("ETAPA 3 - Curadoria do dataset")
    print(f"Destino: {os.path.relpath(config.DIR_CURADO, config.RAIZ)}")

    ausentes = verificar_inventario()
    if ausentes:
        return None

    registros, corrigidos = aplicar_correcao_canal()
    duplicados = detectar_duplicados(registros)
    razoes, nao_observaveis = avaliar_observabilidade(registros)

    motivos = {}
    for numero in duplicados:
        motivos[numero] = "conteudo duplicado"
    if manter_suspeitos:
        print()
        print("   --manter-suspeitos: quarentena por observabilidade DESLIGADA.")
        nao_observaveis = set()
    for numero in nao_observaveis:
        motivos[numero] = "sem assinatura observavel"

    quarentena = set(motivos)
    manifesto = gravar(registros, quarentena, motivos, razoes)

    cabecalho("3.6 RESUMO DA CURADORIA")
    print(f"  T1  selecao de canal corrigida em {len(corrigidos)} arquivo(s)")
    print(f"  T2  inventario completo ({len(config.ENSAIOS)}/"
          f"{len(config.ENSAIOS)})")
    print(f"  T3  {len(duplicados)} duplicado(s) removido(s)")
    print(f"  T9  {len(nao_observaveis)} ensaio(s) sem assinatura em quarentena")
    print()
    print(f"  Dataset curado: {manifesto['totais']['incluidos']} ensaios")
    return manifesto


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manter-suspeitos", action="store_true",
        help="nao poe em quarentena os ensaios sem assinatura observavel (T9)")
    args = parser.parse_args()
    executar(manter_suspeitos=args.manter_suspeitos)
