import numpy as np
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
