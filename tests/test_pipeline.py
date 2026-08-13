import numpy as np
import pytest
from src.pipeline.data_processor import DataProcessor
from src.pipeline.model_builder import TinyMLModelBuilder

def test_window_creation():
    """
    Testa se o fatiamento (windowing) com overlap está criando 
    os tensores da maneira matemática correta.
    """
    dummy_ts = np.arange(1024)
    processor = DataProcessor(window_size=512, overlap=0.5)
    
    X, y = processor.create_windows(dummy_ts, label=1)
    
    # Validação do formato do tensor de entrada
    assert X.shape == (3, 512, 1), f"Shape incorreto: {X.shape}"
    assert y.shape == (3,), f"Label shape incorreto: {y.shape}"
    assert y[0] == 1, "Label incorreto."

def test_decimation_stages():
    """
    A fatoração do fator de decimação deve gerar etapas pequenas,
    cujo produto reconstrói o fator original.
    """
    for factor in (12, 48, 4, 1):
        stages = DataProcessor.decimation_stages(factor)
        assert all(q <= 8 for q in stages), f"Etapa grande demais em {stages}"
        produto = 1
        for q in stages:
            produto *= q
        assert produto == factor, f"{stages} não reconstrói o fator {factor}"


def test_resample_preserves_low_frequency():
    """
    A decimação de 12 kHz para 1 kHz deve preservar um seno de 100 Hz
    (bem abaixo do Nyquist de 500 Hz) em frequência e amplitude.
    """
    processor = DataProcessor(window_size=512, overlap=0.5)
    fs_orig, fs_alvo, freq = 12000, 1000, 100.0
    t = np.arange(0, 1.0, 1.0 / fs_orig)
    sinal = np.sin(2 * np.pi * freq * t)

    decimado = processor.resample_to(sinal, fs_orig, fs_alvo)

    assert len(decimado) == pytest.approx(fs_alvo, rel=0.02)
    # Amplitude preservada (descartando as bordas, onde o filtro tem transiente)
    miolo = decimado[50:-50]
    assert np.max(np.abs(miolo)) == pytest.approx(1.0, abs=0.05)
    # Pico do espectro na frequência correta
    espectro = np.abs(np.fft.rfft(miolo))
    pico_hz = np.fft.rfftfreq(len(miolo), 1.0 / fs_alvo)[np.argmax(espectro)]
    assert pico_hz == pytest.approx(freq, abs=2.0)


def test_split_has_no_shared_samples():
    """
    O corte treino/teste deve deixar uma zona morta de uma janela inteira,
    para que nenhuma amostra de treino reapareça numa janela de teste.
    """
    processor = DataProcessor(window_size=512, overlap=0.9375)
    serie = np.arange(10000, dtype=np.float64)

    treino, teste = processor.split_time_series(serie, test_ratio=0.2)

    assert len(treino) > 0 and len(teste) > 0
    # Como a série é 0..N, o valor da amostra é o próprio índice temporal.
    assert treino[-1] + processor.window_size <= teste[0], "Zona morta ausente"
    assert len(np.intersect1d(treino, teste)) == 0, "Vazamento entre treino e teste"


def test_model_build():
    """
    Testa se o modelo Keras está respeitando as dimensões e 
    consegue ser construído para a classificação.
    """
    builder = TinyMLModelBuilder()
    model = builder.build_cnn1d(input_shape=(512, 1), num_classes=3)
    
    # Validando se o modelo possui a última camada com unidades = num_classes
    assert model.layers[-1].units == 3, "Número de classes incorreto na última camada densa."
    # Validando o input original da primeira convolucional
    assert model.input_shape == (None, 512, 1), "Formato de entrada não correspondente."
