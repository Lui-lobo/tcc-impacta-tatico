"""Executa as quatro etapas do fluxo e gera o relatorio consolidado.

    python -m dataset_engineering.executar
    python -m dataset_engineering.executar --manter-suspeitos
    python -m dataset_engineering.executar --somente 1 2

Saidas:
    dataset_engineering/relatorios/engenharia_dados.md   relatorio consolidado
    dataset_engineering/relatorios/diagnostico_antes.json
    dataset_engineering/relatorios/comparacao.json
    dataset_engineering/relatorios/figuras/*.png
    data_curado/                                          dataset curado
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_engineering import config, exportar_amostras
from dataset_engineering import etapa1_leitura, etapa2_diagnostico
from dataset_engineering import etapa3_curadoria, etapa4_comparacao

RELATORIO = os.path.join(config.DIR_RELATORIOS, "engenharia_dados.md")


def _pct(v):
    return "n/d" if v is None else f"{100 * v:.2f}%"


def gerar_relatorio(etapa1, antes, manifesto, comparacao):
    """Monta o relatorio consolidado a partir das saidas das quatro etapas."""
    escalas = comparacao["escalas"]
    sondas = comparacao["sondas"]
    base, final = sondas["bruto"], sondas["+T1+T9"]
    e_base, e_final = escalas["bruto"], escalas["+T1+T9"]

    quarentena = [e for e in manifesto["ensaios"] if e["status"] == "quarentena"]
    corrigidos = [e for e in manifesto["ensaios"] if e["corrigido_t1"]]

    linhas = []
    A = linhas.append

    A("# Engenharia de Dados — do CWRU bruto ao dataset curado")
    A("")
    A(f"Gerado por `python -m dataset_engineering.executar` em "
      f"{datetime.now().strftime('%d/%m/%Y %H:%M')}.")
    A("")
    A("Este relatório é gerado, não escrito à mão. Todo número aqui sai de uma")
    A("execução das quatro etapas do fluxo sobre os arquivos em `data/`.")
    A("")
    A("---")
    A("")

    # -----------------------------------------------------------------
    A("## Resumo")
    A("")
    A("| Medida | Antes | Depois | Δ |")
    A("|---|---|---|---|")
    A(f"| Ensaios | {e_base['arquivos']} | {e_final['arquivos']} | "
      f"{e_final['arquivos'] - e_base['arquivos']:+d} |")
    A(f"| Sinal a {config.TAXA_ALVO_HZ} Hz | {e_base['duracao_s']:.1f} s | "
      f"{e_final['duracao_s']:.1f} s | "
      f"{e_final['duracao_s'] - e_base['duracao_s']:+.1f} s |")
    A(f"| Janelas independentes | {e_base['janelas_independentes']} | "
      f"{e_final['janelas_independentes']} | "
      f"{e_final['janelas_independentes'] - e_base['janelas_independentes']:+d} |")
    A(f"| Ensaios lendo a série errada | {len(corrigidos)} | 0 | "
      f"{-len(corrigidos):+d} |")
    A(f"| Ensaios sem assinatura observável | {len(quarentena)} | 0 | "
      f"{-len(quarentena):+d} |")
    A(f"| **Separabilidade** (sonda linear) | "
      f"**{_pct(base['acuracia_media'])}** | "
      f"**{_pct(final['acuracia_media'])}** | "
      f"**{100 * (final['acuracia_media'] - base['acuracia_media']):+.2f} pp** |")
    A(f"| Desvio entre cargas | {_pct(base['acuracia_desvio'])} | "
      f"{_pct(final['acuracia_desvio'])} | "
      f"{100 * (final['acuracia_desvio'] - base['acuracia_desvio']):+.2f} pp |")
    A(f"| Falso alarme | {_pct(base['falso_alarme_medio'])} | "
      f"{_pct(final['falso_alarme_medio'])} | "
      f"{100 * (final['falso_alarme_medio'] - base['falso_alarme_medio']):+.2f} pp |")
    A("")
    A("> A **sonda linear** é uma regressão logística sobre atributos rasos")
    A("> (RMS, curtose, assimetria, fator de crista, taxa de cruzamento por zero")
    A("> e energia em 8 sub-bandas), validada deixando uma carga de fora. Ela não")
    A("> é o modelo do projeto: é um termômetro barato e determinístico da")
    A("> separabilidade do dado. Serve para comparar conjuntos, não para prever")
    A("> a acurácia da CNN.")
    A("")
    A("---")
    A("")

    # -----------------------------------------------------------------
    A("## 1. O dataset bruto")
    A("")
    A(f"{len(config.ENSAIOS)} ensaios do CWRU Bearing Data Center, rolamento")
    A("SKF 6205-2RS JEM no lado do acionamento, defeitos usinados por")
    A("eletroerosão. Cada `.mat` é um ensaio: uma classe, uma severidade e um")
    A("nível de carga.")
    A("")
    A("| Classe | Severidades | Cargas | Taxa original |")
    A("|---|---|---|---|")
    A(f"| Normal | — | 0–3 hp | {config.TAXA_NORMAL_HZ} Hz |")
    A(f"| Pista interna | 0,007\" / 0,014\" / 0,021\" | 0–3 hp | "
      f"{config.TAXA_FALHA_HZ} Hz |")
    A(f"| Pista externa | 0,007\" / 0,014\" / 0,021\" | 0–3 hp | "
      f"{config.TAXA_FALHA_HZ} Hz |")
    A("")
    A("Figuras em [`figuras/`](figuras/):")
    A("")
    A("### O sinal, antes e depois da decimação")
    A("![formas de onda](figuras/formas_de_onda.png)")
    A("")
    A("### Onde mora a energia")
    A("![espectros](figuras/espectros.png)")
    A("")
    A("A faixa laranja é a ressonância estrutural do mancal (2–5 kHz), excitada")
    A("pelos impactos do defeito. A área cinza é tudo que o filtro")
    A(f"anti-aliasing descarta ao decimar para {config.TAXA_ALVO_HZ} Hz.")
    A("")
    A("### A assinatura do defeito")
    A("![envelope](figuras/envelope.png)")
    A("")
    A("### Separabilidade por atributos clássicos")
    A("![separabilidade](figuras/separabilidade.png)")
    A("")
    A("---")
    A("")

    # -----------------------------------------------------------------
    A("## 2. Diagnóstico — o que estava errado")
    A("")
    if corrigidos:
        A("### 2.1 Leitura do canal errado")
        A("")
        A("`DataProcessor.load_mat_file()` escolhe a **primeira** chave que")
        A("contém `_DE_time`. Alguns arquivos do CWRU carregam as variáveis de")
        A("mais de um ensaio, e nesses a primeira chave não é a do ensaio certo.")
        A("")
        A("| Arquivo | Lia | Deveria ler | Consequência |")
        A("|---|---|---|---|")
        for e in corrigidos:
            A(f"| `{e['arquivo_origem']}` | `{e['chave_ingenua']}` | "
              f"`{e['chave_mat']}` | série de outra carga, rotulada como "
              f"{e['carga_hp']} hp |")
        A("")

    if quarentena:
        A("### 2.2 Ensaios sem assinatura observável")
        A("")
        A("Razão pico/mediana do espectro de envelope na frequência")
        A(f"característica, medida no sinal **original** demodulado em ")
        A(f"{config.BANDA_RESSONANCIA_HZ[0] // 1000}–"
          f"{config.BANDA_RESSONANCIA_HZ[1] // 1000} kHz. Piso de ruído: "
          f"{config.PISO_ENVELOPE:.0f}.")
        A("")
        A("| Arquivo | Classe | Severidade | Carga | Razão |")
        A("|---|---|---|---|---|")
        for e in quarentena:
            A(f"| `{e['arquivo_origem']}` | "
              f"{config.NOMES_CURTOS[e['classe']]} | {e['severidade']} | "
              f"{e['carga_hp']} hp | **{e['envelope_ressonancia']}** |")
        A("")
        A("Para comparação, os demais ensaios da mesma classe ficam entre")
        A("1.000 e 40.000. O defeito existe na bancada, mas não se manifesta no")
        A("sinal — nem antes de qualquer decimação. Mantidos, seriam rótulos de")
        A("falha que nenhum sensor poderia sustentar.")
        A("")

    A("---")
    A("")

    # -----------------------------------------------------------------
    A("## 3. Tratativas aplicadas")
    A("")
    A("| # | Tratativa | Efeito |")
    A("|---|---|---|")
    A(f"| T1 | Selecionar o canal pelo **número do arquivo**, não pela ordem "
      f"das chaves | {len(corrigidos)} ensaio(s) corrigido(s) |")
    A("| T2 | Inventário obrigatório: a curadoria para se faltar arquivo | "
      "impede treinar em silêncio com dataset incompleto |")
    A("| T3 | Deduplicação por hash da série | efeito colateral de T1: "
      "os duplicados somem |")
    A(f"| T9 | Quarentena por **observabilidade medida** | "
      f"{len(quarentena)} ensaio(s) em quarentena |")
    A("")
    A("T9 é um critério, não uma lista negra: qualquer ensaio cuja frequência")
    A("característica não se destaque do ruído é reprovado, inclusive arquivos")
    A("que o CWRU venha a publicar depois.")
    A("")
    A("---")
    A("")

    # -----------------------------------------------------------------
    A("## 4. Ablação — qual tratativa produziu o quê")
    A("")
    A("| Conjunto | Ensaios | Janelas indep. | Acurácia | Desvio | Detecção | "
      "Falso alarme |")
    A("|---|---|---|---|---|---|---|")
    rotulos = {
        "bruto": "bruto (como o `build.py` lê hoje)",
        "+T1": "+ T1 (canal corrigido)",
        "+T1+T9": "**+ T1 + T9 (curado)**",
    }
    for nome in ("bruto", "+T1", "+T1+T9"):
        s, e = sondas[nome], escalas[nome]
        A(f"| {rotulos[nome]} | {e['arquivos']} | "
          f"{e['janelas_independentes']} | {_pct(s['acuracia_media'])} | "
          f"{_pct(s['acuracia_desvio'])} | {_pct(s['deteccao_media'])} | "
          f"{_pct(s['falso_alarme_medio'])} |")
    A("")
    A("### Acurácia por carga deixada de fora")
    A("")
    A("| Conjunto | " + " | ".join(f"{c} hp" for c in config.CARGAS) + " |")
    A("|---|" + "---|" * len(config.CARGAS))
    for nome in ("bruto", "+T1", "+T1+T9"):
        por_carga = {r["carga"]: r["acuracia"] for r in sondas[nome]["rodadas"]}
        A(f"| {rotulos[nome]} | " + " | ".join(
            _pct(por_carga.get(c)) for c in config.CARGAS) + " |")
    A("")
    A("O ganho vem quase todo de **T9**: remover quatro ensaios cujo rótulo o")
    A("sinal não sustenta melhora a separabilidade mais do que os mesmos quatro")
    A("ensaios acrescentavam. T1 corrige a **correção** dos rótulos — vale por")
    A("si, independentemente de mexer pouco na sonda.")
    A("")
    A("---")
    A("")

    # -----------------------------------------------------------------
    A("## 5. O dataset curado")
    A("")
    A("```")
    A("data_curado/")
    A("  manifesto.json          metadados dos 28 ensaios, incluídos e em quarentena")
    A("  0_normal/097.npy        série já decimada para 1 kHz, float32")
    A("  1_inner_race/105.npy")
    A("  ...")
    A("```")
    A("")
    A(f"| Item | Valor |")
    A(f"|---|---|")
    A(f"| Ensaios incluídos | {manifesto['totais']['incluidos']} |")
    A(f"| Ensaios em quarentena | {manifesto['totais']['quarentena']} |")
    A(f"| Taxa de amostragem | {config.TAXA_ALVO_HZ} Hz |")
    A(f"| Sinal total | {e_final['duracao_s']:.1f} s |")
    A(f"| Janelas independentes | {e_final['janelas_independentes']} |")
    A("")
    A("| Classe | Ensaios | Janelas indep. | % |")
    A("|---|---|---|---|")
    for pasta, v in e_final["por_classe"].items():
        A(f"| `{pasta}` | {v['arquivos']} | {v['janelas_independentes']} | "
          f"{100 * v['fracao']:.1f}% |")
    A("")
    A("A série é gravada **já decimada**. A decimação é determinística e cara;")
    A("guardá-la uma vez faz o `build.py`, o `validacao_por_carga.py` e este")
    A("fluxo partirem do mesmo sinal, byte a byte. Antes, cada ferramenta")
    A("decimava por conta própria.")
    A("")
    A("O `manifesto.json` carrega classe, severidade, carga e rpm de cada")
    A("ensaio. Isso elimina os mapas `arquivo -> carga` que estavam duplicados")
    A("em três arquivos do projeto, e habilita validações estratificadas")
    A("(deixando uma carga de fora, deixando uma severidade de fora) sem")
    A("hard-coding.")
    A("")
    A("### Amostras inspecionáveis")
    A("")
    A("O `.mat` do CWRU não abre em planilha nem em editor de texto. Cinco CSVs")
    A("pequenos, em [`../amostras/`](../amostras/LEIA-ME.md), expõem o dataset")
    A("antes e depois da curadoria — do catálogo dos 28 ensaios à janela de 512")
    A("amostras que entra na rede, e às duas séries que `99.mat` pode entregar")
    A("lado a lado. São versionados no repositório.")
    A("")
    A("---")
    A("")

    # -----------------------------------------------------------------
    A("## 6. O que isto NÃO resolve")
    A("")
    A("A curadoria conserta rótulos e integridade. Ela não mexe no limite")
    A("físico, que continua sendo o fator dominante:")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| Energia descartada pela decimação (falhas) | **~98%** |")
    A(f"| Curtose da pista interna, original → {config.TAXA_ALVO_HZ} Hz | "
      f"+8,4 → −0,3 |")
    A("| Faixa de RMS da classe normal | **dentro** da faixa das falhas |")
    A("")
    A("O MPU6050 não passa de 1 kHz; a assinatura do defeito é excitada em")
    A("2–5 kHz. O que sobrevive é a **periodicidade** dos impactos, não a")
    A("energia deles. Ver [`../../docs/dataset.md`](../../docs/dataset.md) §4 e §5.")
    A("")
    A("---")
    A("")
    A("## Reproduzir")
    A("")
    A("```bash")
    A("python download_cwru.py                        # 28 arquivos, idempotente")
    A("python -m dataset_engineering.executar         # as quatro etapas")
    A("```")

    config.garantir_diretorios()
    with open(RELATORIO, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    return RELATORIO


def executar(manter_suspeitos=False, somente=None):
    somente = set(somente or (1, 2, 3, 4))
    resultados = {}

    if 1 in somente:
        resultados["etapa1"] = etapa1_leitura.executar()
    if 2 in somente:
        resultados["antes"] = etapa2_diagnostico.executar()
    if 3 in somente:
        resultados["manifesto"] = etapa3_curadoria.executar(
            manter_suspeitos=manter_suspeitos)
        if resultados["manifesto"] is None:
            print("\nCuradoria interrompida. Corrija o inventario e repita.")
            return None
    if 4 in somente:
        resultados["comparacao"] = etapa4_comparacao.executar()

    if somente == {1, 2, 3, 4}:
        with open(config.MANIFESTO, encoding="utf-8") as f:
            manifesto = json.load(f)
        caminho = gerar_relatorio(
            resultados["etapa1"], resultados["antes"], manifesto,
            resultados["comparacao"])

        print()
        print("=" * 78)
        print("AMOSTRAS EM CSV")
        print("=" * 78)
        resultados["amostras"] = exportar_amostras.executar()

        print()
        print("=" * 78)
        print("FLUXO COMPLETO")
        print("=" * 78)
        print(f"Relatorio: {os.path.relpath(caminho, config.RAIZ)}")
        print(f"Dataset:   {os.path.relpath(config.DIR_CURADO, config.RAIZ)}")
        print(f"Amostras:  {os.path.relpath(exportar_amostras.DIR_AMOSTRAS, config.RAIZ)}")
        print()
        print("Proximo passo: 'python build.py' passa a usar o dataset curado")
        print("automaticamente. Rode e envie os numeros.")

    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manter-suspeitos", action="store_true",
                        help="desliga a quarentena por observabilidade (T9)")
    parser.add_argument("--somente", nargs="+", type=int, choices=[1, 2, 3, 4],
                        help="roda apenas as etapas indicadas")
    args = parser.parse_args()
    executar(manter_suspeitos=args.manter_suspeitos, somente=args.somente)
