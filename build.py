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


def main():
    print("========================================")
    print(" INICIANDO BUILD E AVALIAÇÃO DO TINYML")
    print("========================================")

    # Semente unica para numpy, random e TensorFlow. Sem isso cada execucao
    # produz um modelo diferente, e os numeros do relatorio nao poderiam ser
    # reproduzidos por quem avaliar o TCC.
    tf.keras.utils.set_random_seed(SEED)

    print("[1/5] Carregando, decimando e janelando os datasets...")
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

    print("\n[2/5] Construindo e treinando a CNN 1D...")
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

    print("\n[3/5] Quantizando (Float32 -> Int8)...")
    tflite_path = os.path.join(PIO_PROJECT_DIR, "model_quantized.tflite")
    quantizer = ModelQuantizer(keras_model=model, representative_dataset=X_train)
    quantizer.quantize_to_int8(output_path=tflite_path)

    print("\n[4/5] Convertendo para artefatos da linguagem C (projeto PlatformIO)...")
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

    print("\n[5/5] Executando Inferências Finais e Medindo Hardware...")
    report_path = "relatorios/relatorio_metricas.md"
    ModelEvaluator.generate_metrics_report(model, history, X_test, y_test, tflite_path, report_path)

    print("\n========================================")
    print("[SUCESSO] Pipeline Finalizado!")
    print(f"Modelo em C:    {c_source_path}")
    print(f"Parametros:     {params_path}")
    print(f"Para gravar:    cd {PIO_PROJECT_DIR} && pio run -t upload")
    print("========================================")

if __name__ == "__main__":
    main()
