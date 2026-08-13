"""Baixa os arquivos do dataset CWRU usados no treino.

Fonte: Case Western Reserve University Bearing Data Center.
https://engineering.case.edu/bearingdatacenter

ESCOLHA DOS ARQUIVOS
--------------------
Todos os arquivos de falha vem do conjunto "12k Drive End Bearing Fault Data",
com o acelerometro montado no mancal do lado do acionamento. Para cada classe de
falha sao baixados os 4 niveis de carga (0 a 3 hp) e os 3 diametros de defeito
(0,007", 0,014" e 0,021").

Essa diversidade e deliberada: com um unico arquivo por classe, o modelo pode
aprender a assinatura de UM defeito especifico sob UMA carga. Com 12, ele
precisa aprender o que ha de comum entre defeitos de severidades e rotacoes
diferentes na mesma pista - que e o que um sistema de manutencao preditiva
realmente precisa reconhecer.

DESEQUILIBRIO INERENTE
----------------------
A classe normal tem apenas 4 arquivos porque o CWRU so disponibiliza 4 ensaios
de baseline (97 a 100), gravados a 48 kHz. Nao ha como equilibrar a contagem sem
descartar dados de falha. O build.py compensa com pesos por classe, e o relatorio
apresenta recall por classe alem da acuracia global.
"""
import os
import sys

import requests

BASE_URL = "https://engineering.case.edu/sites/default/files/"

# Carga (hp) -> rotacao aproximada do eixo.
RPM_POR_CARGA = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# pasta -> {arquivo: descricao}
CATALOGO = {
    # Normal Baseline, 48 kHz.
    "0_normal": {
        "97.mat": "baseline, 0 hp (1797 rpm)",
        "98.mat": "baseline, 1 hp (1772 rpm)",
        "99.mat": "baseline, 2 hp (1750 rpm)",
        "100.mat": "baseline, 3 hp (1730 rpm)",
    },
    # Falha na pista interna, 12 kHz Drive End.
    "1_inner_race": {
        "105.mat": '0,007", 0 hp', "106.mat": '0,007", 1 hp',
        "107.mat": '0,007", 2 hp', "108.mat": '0,007", 3 hp',
        "169.mat": '0,014", 0 hp', "170.mat": '0,014", 1 hp',
        "171.mat": '0,014", 2 hp', "172.mat": '0,014", 3 hp',
        "209.mat": '0,021", 0 hp', "210.mat": '0,021", 1 hp',
        "211.mat": '0,021", 2 hp', "212.mat": '0,021", 3 hp',
    },
    # Falha na pista externa, 12 kHz Drive End, defeito centrado em 6 horas
    # (posicao de maxima carga - a mais usada na literatura).
    "2_outer_race": {
        "130.mat": '0,007", 0 hp', "131.mat": '0,007", 1 hp',
        "132.mat": '0,007", 2 hp', "133.mat": '0,007", 3 hp',
        "197.mat": '0,014", 0 hp', "198.mat": '0,014", 1 hp',
        "199.mat": '0,014", 2 hp', "200.mat": '0,014", 3 hp',
        "234.mat": '0,021", 0 hp', "235.mat": '0,021", 1 hp',
        "236.mat": '0,021", 2 hp', "237.mat": '0,021", 3 hp',
    },
}


def baixar(nome: str, destino: str) -> str:
    """Baixa um arquivo se ainda nao existir. Devolve 'ok', 'existe' ou 'falha'."""
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        return "existe"

    parcial = destino + ".parcial"
    try:
        resposta = requests.get(BASE_URL + nome, stream=True, timeout=120)
        resposta.raise_for_status()
        with open(parcial, "wb") as f:
            for bloco in resposta.iter_content(chunk_size=65536):
                f.write(bloco)
        # Renomeia so no fim: um download interrompido nao deixa para tras um
        # .mat truncado que o scipy tentaria carregar na proxima execucao.
        os.replace(parcial, destino)
        return "ok"
    except Exception as erro:
        if os.path.exists(parcial):
            os.remove(parcial)
        print(f"      FALHA em {nome}: {erro}")
        return "falha"


def main() -> int:
    print("Baixando o dataset CWRU (Case Western Reserve University)")
    print(f"Origem: {BASE_URL}")
    print()

    total = {"ok": 0, "existe": 0, "falha": 0}
    falhados = []

    for pasta, arquivos in CATALOGO.items():
        os.makedirs(os.path.join("data", pasta), exist_ok=True)
        print(f"[{pasta}] {len(arquivos)} arquivos")

        for nome, descricao in arquivos.items():
            destino = os.path.join("data", pasta, nome)
            estado = baixar(nome, destino)
            total[estado] += 1
            if estado == "falha":
                falhados.append(f"{pasta}/{nome}")
                continue
            tamanho = os.path.getsize(destino) / 1024.0
            marca = "novo" if estado == "ok" else "ja existia"
            print(f"      {nome:<10} {descricao:<22} {tamanho:>8.0f} KB  ({marca})")
        print()

    print("-" * 60)
    print(f"Baixados: {total['ok']} | Ja existiam: {total['existe']} | "
          f"Falhas: {total['falha']}")
    if falhados:
        print("Arquivos que falharam (reexecute para tentar de novo):")
        for f in falhados:
            print(f"  {f}")
        return 1

    print("Pronto. Execute 'python build.py' para retreinar com os novos dados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
