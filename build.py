import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from src.pipeline.data_processor import DataProcessor
from src.pipeline.model_builder import TinyMLModelBuilder
from src.pipeline.quantizer import ModelQuantizer
from src.pipeline.c_generator import CArtifactGenerator
from src.pipeline.evaluator import ModelEvaluator

def load_datasets(base_dir="data", window_size=512):
    processor = DataProcessor(window_size=window_size)
    X_list = []
    y_list = []
    
    # Lê as pastas do diretório data (ex: '0_normal', '1_inner_race')
    class_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    print(f"Pastas estruturais encontradas em '{base_dir}': {class_folders}")
    
    for folder in class_folders:
        try:
            class_label = int(folder.split('_')[0]) # Extrai o '0' de '0_normal'
        except ValueError:
            continue
            
        folder_path = os.path.join(base_dir, folder)
        mat_files = glob.glob(os.path.join(folder_path, "*.mat"))
        
        for mat_file in mat_files:
            print(f"  -> Processando arquivo {mat_file} (Classe {class_label})...")
            time_series = processor.load_mat_file(mat_file)
            X, y = processor.create_windows(time_series, label=class_label)
            if len(X) > 0:
                X_list.append(X)
                y_list.append(y)
                
    if not X_list:
        raise ValueError("Nenhum arquivo .mat encontrado nas subpastas.")
        
    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.concatenate(y_list)
    return X_all, y_all, len(class_folders)

def main():
    print("========================================")
    print(" INICIANDO BUILD E AVALIAÇÃO DO TINYML")
    print("========================================")
    
    window_size = 512
    
    print("[1/5] Carregando datasets estruturados...")
    X_all, y_all, num_classes = load_datasets(base_dir="data", window_size=window_size)
    
    # Separa 20% dos dados para validação real e balanceada (stratify)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    
    # Normalização Estatística
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    print(f"      Janelas para Treino: {len(X_train)} | Teste: {len(X_test)} | Classes ativas: {num_classes}")
    
    print("\n[2/5] Construindo e treinando a CNN 1D (Agora para valer)...")
    builder = TinyMLModelBuilder()
    model = builder.build_cnn1d(input_shape=(window_size, 1), num_classes=num_classes)
    
    # Ajustado para 5 épocas, aproveitando que agora os dados têm características reais diferentes
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    
    print("\n[3/5] Quantizando (Float32 -> Int8)...")
    tflite_path = "arduino_deploy/tinyml_esp32/model_quantized.tflite"
    quantizer = ModelQuantizer(keras_model=model, representative_dataset=X_train)
    quantizer.quantize_to_int8(output_path=tflite_path)
    
    print("\n[4/5] Convertendo para artefato da linguagem C...")
    c_header_path = "arduino_deploy/tinyml_esp32/model_tflite.h"
    CArtifactGenerator.tflite_to_c_array(tflite_path, c_header_path)
    
    print("\n[5/5] Executando Inferências Finais e Medindo Hardware...")
    report_path = "relatorios/relatorio_metricas.md"
    ModelEvaluator.generate_metrics_report(model, history, X_test, y_test, tflite_path, report_path)
    
    print("\n========================================")
    print("[SUCESSO] Pipeline Finalizado!")
    print(f"O Arquivo header foi gerado em: {c_header_path}")
    print("========================================")

if __name__ == "__main__":
    main()
