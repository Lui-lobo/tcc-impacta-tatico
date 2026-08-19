"""Leitor do dataset curado produzido pelo fluxo dataset_engineering.

O `data_curado/` guarda cada ensaio como um .npy JA DECIMADO para a taxa de
destino, acompanhado de um manifesto com classe, severidade, carga e rotacao.
Isso resolve dois problemas de uma vez:

1. Toda ferramenta parte do MESMO sinal. Antes, o build.py e o
   validacao_por_carga.py decimavam por conta propria; qualquer divergencia
   entre eles seria indistinguivel de uma diferenca de modelo.

2. Os metadados deixam de ser hard-coded. O mapa `arquivo -> carga` estava
   duplicado em tres lugares do projeto, e o mapa `arquivo -> severidade`, em
   nenhum - o que impedia validar por severidade.

Quando `data_curado/` nao existe, quem chama deve cair de volta na leitura
direta de `data/*.mat`. E o que o build.py faz: o fluxo de curadoria e uma
melhoria opcional, nao um pre-requisito para treinar.
"""
import json
import os
from typing import List, Optional, Tuple

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_PADRAO = os.path.join(RAIZ, "data_curado")


def caminho_manifesto(base_dir: Optional[str] = None) -> str:
    return os.path.join(base_dir or DIR_PADRAO, "manifesto.json")


def disponivel(base_dir: Optional[str] = None) -> bool:
    return os.path.exists(caminho_manifesto(base_dir))


def ler_manifesto(base_dir: Optional[str] = None) -> dict:
    with open(caminho_manifesto(base_dir), encoding="utf-8") as f:
        return json.load(f)


def carregar(base_dir: Optional[str] = None) -> Tuple[dict, List[dict]]:
    """Devolve (manifesto, ensaios). Ensaios em quarentena ficam de fora.

    Cada ensaio traz a serie em `serie`, ja na taxa `fs_hz` do manifesto. O
    campo `taxa_origem` e a taxa em que o ensaio foi gravado no CWRU, mantido
    apenas para os relatorios: a serie devolvida NAO precisa de decimacao.
    """
    base_dir = base_dir or DIR_PADRAO
    manifesto = ler_manifesto(base_dir)

    ensaios = []
    for entrada in manifesto["ensaios"]:
        if entrada["status"] != "incluido":
            continue
        caminho = os.path.join(base_dir, entrada["npy"])
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"{caminho} listado no manifesto mas ausente em disco. "
                f"Rode 'python -m dataset_engineering.executar' de novo.")
        ensaios.append({
            "numero": entrada["numero"],
            "nome": entrada["arquivo_origem"],
            "classe": entrada["classe"],
            "pasta": entrada["pasta"],
            "severidade": entrada["severidade"],
            "carga": entrada["carga_hp"],
            "rpm": entrada["rpm_nominal"],
            "taxa_origem": entrada["fs_origem_hz"],
            "fs": entrada["fs_hz"],
            "serie": np.load(caminho).astype(np.float64),
        })

    ensaios.sort(key=lambda e: (e["classe"], e["numero"]))
    return manifesto, ensaios


def classes_presentes(ensaios: List[dict]) -> List[str]:
    """Pastas de classe representadas, em ordem de rotulo."""
    vistas = {}
    for e in ensaios:
        vistas[e["classe"]] = e["pasta"]
    return [vistas[k] for k in sorted(vistas)]


def resumo(manifesto: dict, ensaios: List[dict]) -> str:
    amostras = sum(len(e["serie"]) for e in ensaios)
    fs = manifesto["fs_hz"]
    quarentena = manifesto["totais"]["quarentena"]
    return (f"dataset curado v{manifesto['versao']}: {len(ensaios)} ensaios "
            f"({quarentena} em quarentena), {amostras / fs:.1f} s a {fs} Hz, "
            f"gerado em {manifesto['gerado_em']}")
