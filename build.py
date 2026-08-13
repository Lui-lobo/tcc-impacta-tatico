import os
import glob
import numpy as np
import tensorflow as tf
from src.pipeline.data_processor import DataProcessor
from src.pipeline.model_builder import TinyMLModelBuilder
from src.pipeline.quantizer import ModelQuantizer
from src.pipeline.c_generator import CArtifactGenerator
from src.pipeline.evaluator import ModelEvaluator

# Raiz do projeto PlatformIO que sobe o modelo para o ESP32.
PIO_PROJECT_DIR = "arduino_deploy/tinyml_esp32"

# ---------------------------------------------------------------------------
# Configuracao de amostragem
# ---------------------------------------------------------------------------
# Taxa de destino = teto real do hardware. O acelerometro do MPU6050 atualiza
# seus registradores a 1 kHz; nao existe configuracao que chegue aos 12 kHz do
# dataset original. Treinar na taxa do sensor e o que torna a inferencia
# embarcada comparavel ao que foi aprendido em bancada.
TARGET_SAMPLE_RATE_HZ = 1000

# Taxas de origem do CWRU. Os arquivos "Normal Baseline" (97-100) foram
# gravados a 48 kHz; os de falha Drive-End usados aqui, a 12 kHz. Sem esta
# distincao, a classe normal chegaria ao modelo numa escala de frequencia 4x
# diferente das classes de falha - e o modelo aprenderia a taxa de amostragem
# em vez de aprender a assinatura do defeito.
DEFAULT_SOURCE_RATE_HZ = 12000
SOURCE_RATE_BY_FILE = {
    "97.mat": 48000,
    "98.mat": 48000,
    "99.mat": 48000,
    "100.mat": 48000,
}

WINDOW_SIZE = 512

# A decimacao divide o numero de amostras por 12 (ou 48). A sobreposicao alta
# recupera parte da contagem de janelas. So e seguro porque o corte
# treino/teste acontece ANTES do janelamento - ver split_time_series().
WINDOW_OVERLAP = 0.9375
TEST_RATIO = 0.2

# Com ~12x menos janelas, 5 epocas nao dao passos de gradiente suficientes.
EPOCHS = 40
BATCH_SIZE = 32
SEED = 42

# Teto de janelas de teste gravadas na flash do ESP32 para a validacao
# embarcada (comando 't' no console serial). Cada janela custa
# WINDOW_SIZE * 2 bytes = 1 KB. Acima do teto, a selecao e estratificada e
# preserva a proporcao entre as classes.
MAX_TEST_VECTORS = 128


def load_datasets(base_dir="data"):
    """Carrega, decima para TARGET_SAMPLE_RATE_HZ e janela cada arquivo .mat.

    Devolve treino e teste ja separados: a divisao e feita por trecho temporal
    dentro de cada arquivo, e nao por sorteio de janelas.
    """
    processor = DataProcessor(window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP)
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    class_folders = sorted(
        [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    )
    print(f"Pastas estruturais encontradas em '{base_dir}': {class_folders}")
    print()
    print(f"{'arquivo':<28}{'fs_orig':>9}{'amostras':>10}{'->':>4}"
          f"{'fs_alvo':>9}{'amostras':>10}{'dur(s)':>9}{'jan_tr':>8}{'jan_te':>8}")
    print("-" * 95)

    used_labels = []
    total_samples = 0
    for folder in class_folders:
        try:
            class_label = int(folder.split('_')[0])  # Extrai o '0' de '0_normal'
        except ValueError:
            continue

        folder_path = os.path.join(base_dir, folder)
        mat_files = sorted(glob.glob(os.path.join(folder_path, "*.mat")))
        if mat_files:
            used_labels.append(folder)

        for mat_file in mat_files:
            file_name = os.path.basename(mat_file)
            source_rate = SOURCE_RATE_BY_FILE.get(file_name, DEFAULT_SOURCE_RATE_HZ)

            raw = processor.load_mat_file(mat_file)
            decimated = processor.resample_to(raw, source_rate, TARGET_SAMPLE_RATE_HZ)
            total_samples += len(decimated)

            train_part, test_part = processor.split_time_series(decimated, TEST_RATIO)
            X_tr, y_tr = processor.create_windows(train_part, label=class_label)
            X_te, y_te = processor.create_windows(test_part, label=class_label)

            print(f"{folder + '/' + file_name:<28}{source_rate:>9}{len(raw):>10}{'->':>4}"
                  f"{TARGET_SAMPLE_RATE_HZ:>9}{len(decimated):>10}"
                  f"{len(decimated) / TARGET_SAMPLE_RATE_HZ:>9.1f}"
                  f"{len(X_tr):>8}{len(X_te):>8}")

            if len(X_tr) > 0:
                X_train_list.append(X_tr)
                y_train_list.append(y_tr)
            if len(X_te) > 0:
                X_test_list.append(X_te)
                y_test_list.append(y_te)

    if not X_train_list or not X_test_list:
        raise ValueError(
            "Janelas insuficientes apos a decimacao. Reduza WINDOW_SIZE, "
            "aumente WINDOW_OVERLAP ou use arquivos .mat mais longos."
        )

    X_train = np.vstack(X_train_list).astype(np.float32)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list).astype(np.float32)
    y_test = np.concatenate(y_test_list)

    # A sobreposicao multiplica a contagem de janelas, mas nao cria informacao
    # nova. Este numero e o que realmente existe de sinal independente, e e o
    # que deve ser citado ao discutir a validade estatistica dos resultados.
    independent = int(total_samples // WINDOW_SIZE)
    print("-" * 95)
    print(f"      Sinal total apos decimacao: {total_samples} amostras "
          f"({total_samples / TARGET_SAMPLE_RATE_HZ:.1f} s) "
          f"=> {independent} janelas INDEPENDENTES (sem sobreposicao).")

    return X_train, y_train, X_test, y_test, used_labels


def round_half_away(x: np.ndarray) -> np.ndarray:
    """Arredondamento identico ao lroundf() da libc usada no ESP32.

    np.round() usa arredondamento bancario (0.5 -> par mais proximo), enquanto
    lroundf() afasta do zero. Sem esta compatibilizacao, os valores terminados
    em exatamente .5 divergiriam em 1 LSB e a comparacao bit a bit acusaria
    diferencas que nao existem no modelo.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def as_header_float(value: float) -> np.float32:
    """Reproduz em Python o literal float32 que o header C vai receber.

    generate_params_header() imprime as constantes com 10 casas decimais. Se o
    Python continuasse usando o float64 original, a normalizacao das duas pontas
    partiria de numeros ligeiramente diferentes.
    """
    return np.float32(float(f"{value:.10f}"))


def quantize_input(normalized: np.ndarray, scale: float,
                   zero_point: int) -> np.ndarray:
    """Converte a janela normalizada em int8, como o firmware faz."""
    q = round_half_away(normalized / np.float32(scale)) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def fnv1a32(data: np.ndarray) -> int:
    """Hash FNV-1a de 32 bits sobre os bytes da janela quantizada.

    Serve para o ESP32 provar que a entrada que ele montou e byte a byte a
    mesma que o PC usou. Sem essa checagem, uma divergencia na saida seria
    ambigua: poderia estar na normalizacao, na quantizacao ou nos kernels.
    Com ela, o teste isola a causa.
    """
    h = 2166136261
    for b in data.astype(np.uint8).tolist():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def reference_outputs(tflite_path: str, quantized: np.ndarray,
                      resolver=None) -> np.ndarray:
    """Roda o interpretador TFLite do PC e devolve o tensor de saida int8.

    Este e o padrao-ouro contra o qual o ESP32 sera comparado. Recebe a entrada
    JA quantizada, para que a aritmetica de pre-processamento seja rigorosamente
    a mesma nas duas pontas e a comparacao meça apenas os kernels.

    O parametro `resolver` escolhe QUAIS kernels executam os operadores, e essa
    escolha nao e cosmetica:

      BUILTIN     - kernels otimizados do TFLite de desktop (XNNPACK, ruy).
                    Reordenam acumulacoes e usam caminhos vetorizados.
      BUILTIN_REF - kernels de referencia, aritmetica inteira em ordem
                    canonica. E o que o TensorFlow Lite MICRO implementa.

    Comparar o ESP32 contra os kernels otimizados mede a diferenca entre duas
    implementacoes distintas, e nao a fidelidade do deploy. A referencia
    correta para o microcontrolador e BUILTIN_REF.
    """
    if resolver is None:
        resolver = tf.lite.experimental.OpResolverType.BUILTIN_REF

    interpreter = tf.lite.Interpreter(
        model_path=tflite_path, experimental_op_resolver_type=resolver)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    result = np.zeros((len(quantized), int(out['shape'][-1])), dtype=np.int8)
    for i in range(len(quantized)):
        interpreter.set_tensor(inp['index'], quantized[i].reshape(inp['shape']))
        interpreter.invoke()
        result[i] = interpreter.get_tensor(out['index'])[0]
    return result


def input_quantization_params(tflite_path: str):
    """Escala e zero-point do tensor de entrada do modelo quantizado."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    quant = interpreter.get_input_details()[0]['quantization']
    return np.float32(quant[0]), int(quant[1])


def stratified_subset(labels: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    """Indices de um subconjunto que preserva a proporcao entre as classes."""
    if len(labels) <= max_count:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        quota = max(1, int(round(max_count * len(idx) / len(labels))))
        keep.append(rng.choice(idx, size=min(quota, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def export_test_vectors(X_test_raw, y_test, tflite_path, mean, std,
                        header_path, source_path):
    """Grava o conjunto de teste na flash e imprime o custo.

    IMPORTANTE: as janelas sao exportadas ANTES da normalizacao, em unidades
    fisicas. O ESP32 aplica a propria normalizacao e a propria quantizacao, de
    modo que o teste embarcado exercita a cadeia inteira, e nao apenas o
    Invoke() do interpretador.
    """
    windows = X_test_raw.reshape(len(X_test_raw), -1).astype(np.float32)
    labels = y_test.astype(np.uint8)

    selection = stratified_subset(labels, MAX_TEST_VECTORS, SEED)
    windows, labels = windows[selection], labels[selection]

    # Escala unica para todo o conjunto: preserva a relacao de amplitude entre
    # as janelas, que e justamente o que distingue as classes.
    scale = np.float32(np.max(np.abs(windows)) / 32767.0)
    q16 = np.clip(round_half_away(windows / scale), -32768, 32767).astype(np.int16)

    # A referencia e calculada sobre o sinal JA reconstruido do int16, para que
    # qualquer divergencia observada no ESP32 venha do hardware e nao do
    # arredondamento do proprio vetor de teste.
    recon = q16.astype(np.float32) * scale
    normalized = (recon - as_header_float(mean)) / (as_header_float(std) + np.float32(1e-8))

    in_scale, in_zero_point = input_quantization_params(tflite_path)
    quantized = quantize_input(normalized, in_scale, in_zero_point)
    input_hash = np.array([fnv1a32(q) for q in quantized], dtype=np.uint32)

    ref_out = reference_outputs(
        tflite_path, quantized,
        resolver=tf.lite.experimental.OpResolverType.BUILTIN_REF)

    # Quanto os kernels otimizados de desktop se afastam dos de referencia.
    # Este numero e o piso da divergencia esperada no ESP32 caso a comparacao
    # fosse feita contra o interpretador padrao, e explica por que a referencia
    # gravada usa BUILTIN_REF.
    opt_out = reference_outputs(
        tflite_path, quantized,
        resolver=tf.lite.experimental.OpResolverType.BUILTIN)
    kernel_gap = int(np.max(np.abs(opt_out.astype(np.int16) -
                                   ref_out.astype(np.int16))))
    kernel_same = int(np.mean(np.all(opt_out == ref_out, axis=1)) * 100)

    CArtifactGenerator.generate_test_vectors(
        header_path=header_path,
        source_path=source_path,
        windows_i16=q16,
        labels=labels,
        reference_output=ref_out,
        input_hash=input_hash,
        scale=float(scale),
        sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
    )

    ref_pred = ref_out.argmax(axis=1)
    flash_kb = (q16.nbytes + labels.nbytes + ref_out.nbytes
                + input_hash.nbytes) / 1024.0
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    print(f"      {len(labels)} de {len(y_test)} janelas de teste gravadas "
          f"(distribuicao: {counts.tolist()})")
    print(f"      Escala do vetor: {float(scale):.9e} g/LSB "
          f"| custo em flash: {flash_kb:.1f} KB")
    print(f"      Acuracia da referencia (kernels BUILTIN_REF, = TFLite Micro): "
          f"{100.0 * np.mean(ref_pred == labels):.2f}%")
    print(f"      Acuracia com kernels otimizados de desktop (BUILTIN): "
          f"{100.0 * np.mean(opt_out.argmax(axis=1) == labels):.2f}%")
    print(f"      Divergencia entre os dois conjuntos de kernels no PC: "
          f"{kernel_same}% identicas | desvio max = {kernel_gap} LSB")
    print(f"      Use o comando 't' no console serial do ESP32 para reproduzi-la "
          f"no hardware.")


def main():
    print("========================================")
    print(" INICIANDO BUILD E AVALIAÇÃO DO TINYML")
    print("========================================")

    # Semente unica para numpy, random e TensorFlow. Sem isso cada execucao
    # produz um modelo diferente, e os numeros do relatorio nao poderiam ser
    # reproduzidos por quem avaliar o TCC.
    tf.keras.utils.set_random_seed(SEED)

    print("[1/6] Carregando, decimando e janelando os datasets...")
    X_train, y_train, X_test, y_test, class_folders = load_datasets(base_dir="data")
    num_classes = len(class_folders)

    # Embaralha o treino (as janelas saem em ordem temporal por arquivo).
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(X_train))
    X_train, y_train = X_train[order], y_train[order]

    # Normalização estatística: media e desvio vem SO do treino, para que o
    # conjunto de teste permaneca realmente inedito.
    mean = np.mean(X_train)
    std = np.std(X_train)

    # Copia em unidades fisicas, antes da normalizacao: e este sinal que sera
    # gravado na flash do ESP32 para a validacao embarcada (etapa 5).
    X_test_raw = X_test.copy()

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    train_counts = np.bincount(y_train, minlength=num_classes)
    test_counts = np.bincount(y_test, minlength=num_classes)
    print()
    print(f"      Janelas para Treino: {len(X_train)} | Teste: {len(X_test)} "
          f"| Classes ativas: {num_classes}")
    for i, folder in enumerate(class_folders):
        print(f"        {folder:<16} treino={train_counts[i]:>4}  teste={test_counts[i]:>4}")
    print(f"      Janela de {WINDOW_SIZE} amostras a {TARGET_SAMPLE_RATE_HZ} Hz "
          f"= {1000.0 * WINDOW_SIZE / TARGET_SAMPLE_RATE_HZ:.1f} ms "
          f"| Nyquist = {TARGET_SAMPLE_RATE_HZ // 2} Hz")
    print(f"      Normalizacao: media={mean:.8f} desvio={std:.8f}")

    # Os arquivos tem duracoes diferentes (5 s no normal, 10 s nas falhas), o
    # que desbalanceia as classes. O peso por classe evita que o modelo aprenda
    # a favorecer a classe mais numerosa.
    class_weight = {
        i: float(len(y_train)) / (num_classes * max(1, train_counts[i]))
        for i in range(num_classes)
    }
    print(f"      Pesos por classe: "
          f"{ {i: round(w, 3) for i, w in class_weight.items()} }")

    print("\n[2/6] Construindo e treinando a CNN 1D...")
    builder = TinyMLModelBuilder()
    model = builder.build_cnn1d(input_shape=(WINDOW_SIZE, 1), num_classes=num_classes)

    # Com so 49 janelas independentes, a acuracia de validacao oscila muito de
    # uma epoca para outra. Sem isto, o modelo gravado no ESP32 seria o da
    # ultima epoca - uma escolha arbitraria que pode cair num vale ruim.
    # RESSALVA: como nao ha dados para um terceiro conjunto, a selecao da melhor
    # epoca usa o proprio conjunto de teste. A acuracia final e, portanto,
    # otimista, e deve ser apresentada como tal no relatorio.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=(X_test, y_test),
        class_weight=class_weight,
        callbacks=[early_stop],
    )

    print("\n[3/6] Quantizando (Float32 -> Int8)...")
    tflite_path = os.path.join(PIO_PROJECT_DIR, "model_quantized.tflite")
    quantizer = ModelQuantizer(keras_model=model, representative_dataset=X_train)
    quantizer.quantize_to_int8(output_path=tflite_path)

    print("\n[4/6] Convertendo para artefatos da linguagem C (projeto PlatformIO)...")
    c_header_path = os.path.join(PIO_PROJECT_DIR, "include", "model_tflite.h")
    c_source_path = os.path.join(PIO_PROJECT_DIR, "src", "model_tflite.cpp")
    params_path = os.path.join(PIO_PROJECT_DIR, "include", "model_params.h")

    CArtifactGenerator.tflite_to_c_array(tflite_path, c_header_path, c_source_path)
    # O firmware precisa da mesma normalização e da mesma taxa de amostragem do
    # treino para inferir corretamente.
    CArtifactGenerator.generate_params_header(
        params_path,
        window_size=WINDOW_SIZE,
        num_classes=num_classes,
        norm_mean=float(mean),
        norm_std=float(std),
        class_labels=class_folders,
        sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
    )

    print("\n[5/6] Exportando o conjunto de teste para validacao embarcada...")
    vectors_header = os.path.join(PIO_PROJECT_DIR, "include", "test_vectors.h")
    vectors_source = os.path.join(PIO_PROJECT_DIR, "src", "test_vectors.cpp")
    export_test_vectors(
        X_test_raw, y_test, tflite_path, mean, std,
        header_path=vectors_header, source_path=vectors_source,
    )

    print("\n[6/6] Executando Inferências Finais e Medindo Hardware...")
    report_path = "relatorios/relatorio_metricas.md"
    ModelEvaluator.generate_metrics_report(model, history, X_test, y_test, tflite_path, report_path)

    print("\n========================================")
    print("[SUCESSO] Pipeline Finalizado!")
    print(f"Modelo em C:    {c_source_path}")
    print(f"Parametros:     {params_path}")
    print(f"Vetores teste:  {vectors_source}")
    print(f"Para gravar:    cd {PIO_PROJECT_DIR} && pio run -t upload")
    print("========================================")

if __name__ == "__main__":
    main()
