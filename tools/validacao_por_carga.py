"""Validacao cruzada deixando uma CARGA de fora (leave-one-load-out).

MOTIVACAO
---------
O build.py divide treino e teste por trecho temporal DENTRO de cada arquivo:
os primeiros 80% de cada gravacao vao para o treino e os ultimos 20% para o
teste. Isso elimina o vazamento no nivel da amostra - a zona morta de uma janela
inteira garante que nenhuma amostra aparece dos dois lados - mas nao elimina o
vazamento no nivel da GRAVACAO.

Cada janela de teste vem do mesmo rolamento, com o mesmo defeito, sob a mesma
carga, na mesma montagem e na mesma sessao de gravacao que janelas de treino. O
modelo pode acertar reconhecendo a assinatura daquela gravacao especifica, e nao
o tipo de falha. E por isso que a acuracia desse protocolo passa de 99%.

Este script mede o que interessa para manutencao preditiva: o modelo generaliza
para uma condicao de operacao que nunca viu? Treina com 3 niveis de carga e
testa no quarto, repetindo para as 4 cargas.

PROTOCOLO
---------
- Teste  : todos os arquivos de uma carga, em janelas SEM sobreposicao (janelas
           independentes, para que a acuracia tenha significado estatistico).
- Treino : os primeiros 85% de cada arquivo das outras tres cargas.
- Validacao: os ultimos 15% desses mesmos arquivos, apenas para o early
           stopping. Escolher a melhor epoca no conjunto de teste seria o mesmo
           vazamento por outra porta.

A validacao sai de um trecho temporal e nao de uma quarta carga: sacrificar uma
carga inteira deixaria so duas no treino, o que subestimaria o modelo por falta
de diversidade e nao por incapacidade de generalizar.

USO
---
    python tools/validacao_por_carga.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

import build as B
from src.pipeline.data_processor import DataProcessor
from src.pipeline.model_builder import TinyMLModelBuilder

# arquivo -> carga em hp. Extraido do catalogo do download_cwru.py.
CARGA_POR_ARQUIVO = {}
for _carga, _arquivos in enumerate([
    ["97", "105", "169", "209", "130", "197", "234"],   # 0 hp, 1797 rpm
    ["98", "106", "170", "210", "131", "198", "235"],   # 1 hp, 1772 rpm
    ["99", "107", "171", "211", "132", "199", "236"],   # 2 hp, 1750 rpm
    ["100", "108", "172", "212", "133", "200", "237"],  # 3 hp, 1730 rpm
]):
    for _n in _arquivos:
        CARGA_POR_ARQUIVO[f"{_n}.mat"] = _carga

RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


VALIDACAO_RATIO = 0.15


def carregar_series(base_dir="data"):
    """Devolve {carga: [(serie_decimada, rotulo), ...]}.

    Guarda as series inteiras em vez de janelas prontas: o janelamento depende
    de qual carga esta sendo testada, e refazer a decimacao a cada rodada
    custaria 4x mais tempo.
    """
    proc = DataProcessor(window_size=B.WINDOW_SIZE, overlap=B.WINDOW_OVERLAP)
    series = {c: [] for c in RPM}

    for pasta in sorted(os.listdir(base_dir)):
        caminho = os.path.join(base_dir, pasta)
        if not os.path.isdir(caminho):
            continue
        try:
            rotulo = int(pasta.split("_")[0])
        except ValueError:
            continue

        for arquivo in sorted(os.listdir(caminho)):
            if not arquivo.endswith(".mat"):
                continue
            carga = CARGA_POR_ARQUIVO.get(arquivo)
            if carga is None:
                print(f"[AVISO] {arquivo} nao esta no mapa de cargas; ignorado.")
                continue

            taxa = B.SOURCE_RATE_BY_FILE.get(arquivo, B.DEFAULT_SOURCE_RATE_HZ)
            bruto = proc.load_mat_file(os.path.join(caminho, arquivo))
            series[carga].append((proc.resample_to(bruto, taxa, B.TARGET_SAMPLE_RATE_HZ),
                                  rotulo))
    return series


def montar_conjuntos(series, carga_teste):
    """Janela as series conforme o protocolo, para uma carga de teste."""
    sobreposto = DataProcessor(window_size=B.WINDOW_SIZE, overlap=B.WINDOW_OVERLAP)
    independente = DataProcessor(window_size=B.WINDOW_SIZE, overlap=0.0)

    X_tr, y_tr, X_va, y_va = [], [], [], []
    for carga, itens in series.items():
        if carga == carga_teste:
            continue
        for serie, rotulo in itens:
            treino, validacao = sobreposto.split_time_series(serie, VALIDACAO_RATIO)
            for parte, dX, dy in ((treino, X_tr, y_tr), (validacao, X_va, y_va)):
                janelas, rotulos = sobreposto.create_windows(parte, label=rotulo)
                if len(janelas):
                    dX.append(janelas)
                    dy.append(rotulos)

    X_te, y_te = [], []
    for serie, rotulo in series[carga_teste]:
        janelas, rotulos = independente.create_windows(serie, label=rotulo)
        if len(janelas):
            X_te.append(janelas)
            y_te.append(rotulos)

    return (np.concatenate(X_tr), np.concatenate(y_tr),
            np.concatenate(X_va), np.concatenate(y_va),
            np.concatenate(X_te), np.concatenate(y_te))


def treinar_e_avaliar(series, carga_teste, num_classes, verbose=0):
    X_train, y_train, X_val, y_val, X_test, y_test = montar_conjuntos(
        series, carga_teste)
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    rng = np.random.default_rng(B.SEED)
    ordem = rng.permutation(len(X_train))
    X_train, y_train = X_train[ordem], y_train[ordem]

    media, desvio = np.mean(X_train), np.std(X_train)
    X_train = (X_train - media) / (desvio + 1e-8)
    X_val = (X_val - media) / (desvio + 1e-8)
    X_test = (X_test - media) / (desvio + 1e-8)

    contagem = np.bincount(y_train, minlength=num_classes)
    pesos = {i: float(len(y_train)) / (num_classes * max(1, contagem[i]))
             for i in range(num_classes)}

    tf.keras.utils.set_random_seed(B.SEED)
    modelo = TinyMLModelBuilder().build_cnn1d(
        input_shape=(B.WINDOW_SIZE, 1), num_classes=num_classes)
    modelo.fit(
        X_train, y_train,
        epochs=B.EPOCHS, batch_size=B.BATCH_SIZE, verbose=verbose,
        validation_data=(X_val, y_val), class_weight=pesos,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=0)],
    )

    preditos = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
    acuracia = float(np.mean(preditos == y_test))
    recalls = [float(np.mean(preditos[y_test == c] == c)) if np.any(y_test == c) else float("nan")
               for c in range(num_classes)]

    # Separa as duas tarefas embutidas no problema. Confundir pista interna com
    # externa e um erro de diagnostico; classificar um rolamento defeituoso como
    # normal e um erro de DETECCAO, que num sistema de manutencao preditiva tem
    # consequencia completamente diferente.
    falhas = y_test > 0
    normais = y_test == 0
    deteccao = float(np.mean(preditos[falhas] > 0)) if np.any(falhas) else float("nan")
    falso_alarme = float(np.mean(preditos[normais] > 0)) if np.any(normais) else float("nan")

    return acuracia, recalls, deteccao, falso_alarme, len(y_test), len(y_train)


def main():
    print("Validacao cruzada por carga (leave-one-load-out)")
    print("Teste em janelas independentes, sem sobreposicao.")
    print()

    series = carregar_series()
    num_classes = int(max(r for itens in series.values() for _, r in itens)) + 1

    print(f"{'teste':>7} {'rpm':>6} {'jan_te':>7} {'acuracia':>9} "
          f"{'deteccao':>9} {'f.alarme':>9}   recall por classe")
    print("-" * 82)

    acuracias, deteccoes = [], []
    for carga in sorted(RPM):
        acc, recalls, det, fa, n_te, _ = treinar_e_avaliar(series, carga, num_classes)
        acuracias.append(acc)
        deteccoes.append(det)
        r = "  ".join(f"{v * 100:5.1f}%" for v in recalls)
        print(f"{carga:>4} hp {RPM[carga]:>6} {n_te:>7} {acc * 100:>8.2f}% "
              f"{det * 100:>8.2f}% {fa * 100:>8.2f}%   {r}")

    print("-" * 82)
    print(f"Acuracia de 3 classes (diagnostico): "
          f"{np.mean(acuracias) * 100:.2f}% (desvio {np.std(acuracias) * 100:.2f} pp)")
    print(f"Deteccao de falha (falha x normal):  "
          f"{np.mean(deteccoes) * 100:.2f}% (desvio {np.std(deteccoes) * 100:.2f} pp)")
    print()
    print("Estas medidas valem para uma condicao de operacao INEDITA. A acuracia")
    print("do relatorio principal usa divisao temporal dentro de cada arquivo,")
    print("entao toda janela de teste tem uma contraparte de treino da mesma")
    print("gravacao - e sistematicamente mais otimista.")
    print()
    print("'deteccao' = fracao das janelas com falha que NAO foram chamadas de")
    print("normal. 'f.alarme' = fracao das janelas normais chamadas de falha.")


if __name__ == "__main__":
    main()
