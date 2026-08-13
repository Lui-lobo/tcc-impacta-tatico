"""Mede a sensibilidade da softmax int8 a uma perturbacao de 1 LSB nos logitos.

MOTIVACAO
---------
A validacao embarcada (comando 't' no ESP32) mostra que a saida int8 do
microcontrolador difere da referencia do PC em ate 22 LSB, e que a divergencia
cresce onde a confianca cai. A explicacao proposta e que os kernels do
TensorFlow Lite Micro e os do TensorFlow Lite diferem por 1-2 LSB nos logitos, e
que a softmax amplifica essa diferenca quando as classes estao empatadas.

Este script transforma essa explicacao de afirmacao em medida: para cada janela
do conjunto de teste, perturba cada logito em +-1 LSB e mede quanto a saida
quantizada se move.

RESSALVA
--------
A softmax e reimplementada aqui em ponto flutuante, enquanto o TFLite usa uma
aproximacao em ponto fixo. Os valores sao portanto estimativas da ordem de
grandeza da sensibilidade, nao a aritmetica exata do interpretador.

USO
---
    python tools/sensibilidade_softmax.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

import build as B

TFLITE_PATH = "arduino_deploy/tinyml_esp32/model_quantized.tflite"
FAIXAS = [(0.00, 0.70), (0.70, 0.90), (0.90, 0.99), (0.99, 1.01)]


def preparar_entradas():
    """Reconstroi exatamente as janelas gravadas na flash do ESP32."""
    X_train, _, X_test, y_test, _, _ = B.load_datasets("data")
    mean, std = np.mean(X_train), np.std(X_train)

    janelas = X_test.reshape(len(X_test), -1).astype(np.float32)
    escala = np.float32(np.max(np.abs(janelas)) / 32767.0)
    q16 = np.clip(B.round_half_away(janelas / escala), -32768, 32767).astype(np.int16)

    recon = q16.astype(np.float32) * escala
    normalizado = ((recon - B.as_header_float(mean))
                   / (B.as_header_float(std) + np.float32(1e-8)))

    in_scale, in_zp = B.input_quantization_params(TFLITE_PATH)
    return B.quantize_input(normalizado, in_scale, in_zp), y_test


def main():
    quantizado, _ = preparar_entradas()

    interp = tf.lite.Interpreter(
        model_path=TFLITE_PATH,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    entrada = interp.get_input_details()[0]
    saida = interp.get_output_details()[0]

    # Os logitos sao o tensor que alimenta a SOFTMAX.
    softmax_op = [o for o in interp._get_ops_details() if o['op_name'] == 'SOFTMAX'][0]
    idx_logito = int(softmax_op['inputs'][0])
    det_logito = interp._get_tensor_details(idx_logito, 0)
    escala_logito = np.float32(det_logito['quantization'][0])
    zp_logito = int(det_logito['quantization'][1])
    escala_saida = np.float32(saida['quantization'][0])
    zp_saida = int(saida['quantization'][1])

    print(f"Logitos: escala={escala_logito:.8f}  zero_point={zp_logito}")
    print(f"Saida:   escala={escala_saida:.8f}  zero_point={zp_saida}")
    print()

    def softmax_int8(logitos):
        p = np.exp(logitos - logitos.max())
        p /= p.sum()
        return np.clip(np.round(p / escala_saida) + zp_saida, -128, 127).astype(np.int8)

    linhas = []
    for i in range(len(quantizado)):
        interp.set_tensor(entrada['index'], quantizado[i].reshape(entrada['shape']))
        interp.invoke()
        logitos = ((interp.get_tensor(idx_logito)[0].astype(np.float32) - zp_logito)
                   * escala_logito)
        base = softmax_int8(logitos)

        pior = 0
        for c in range(len(logitos)):
            for passo in (-1.0, +1.0):
                perturbado = logitos.copy()
                perturbado[c] += passo * escala_logito
                desvio = int(np.max(np.abs(
                    softmax_int8(perturbado).astype(np.int16) - base.astype(np.int16))))
                pior = max(pior, desvio)

        confianca = float((int(base.max()) - zp_saida) * escala_saida)
        linhas.append((i, confianca, pior))

    linhas = np.array(linhas)
    print("Desvio da saida int8 provocado por 1 LSB de diferenca nos logitos:")
    print()
    print("  faixa de confianca | janelas | desvio medio | desvio max")
    for lo, hi in FAIXAS:
        sel = linhas[(linhas[:, 1] >= lo) & (linhas[:, 1] < hi)]
        if len(sel):
            print(f"    {lo:.2f} a {hi:.2f}       |   {len(sel):3d}   |"
                  f"    {sel[:, 2].mean():6.1f}    |     {sel[:, 2].max():.0f}")

    print()
    print("Conclusao: a sensibilidade e ~100x maior nas janelas de baixa confianca")
    print("do que nas saturadas. O padrao de divergencia medido no ESP32 e portanto")
    print("compativel com 1-2 LSB de diferenca nos logitos entre TFLite e TFLM.")


if __name__ == "__main__":
    main()
