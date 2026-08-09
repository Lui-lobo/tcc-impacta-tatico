import numpy as np
import scipy.io
from typing import Tuple

class DataProcessor:
    def __init__(self, window_size: int = 512, overlap: float = 0.5):
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))

    def load_mat_file(self, file_path: str) -> np.ndarray:
        try:
            mat_data = scipy.io.loadmat(file_path)
        except Exception as e:
            raise IOError(f"Falha ao carregar {file_path}: {e}")
            
        de_key = next((key for key in mat_data.keys() if '_DE_time' in key), None)
        
        if not de_key:
            # Fallback iterativo caso o CWRU esteja com outra nomenclatura
            arrays = {k: v for k, v in mat_data.items() if not k.startswith('__') and isinstance(v, np.ndarray)}
            if not arrays:
                raise ValueError("Nenhum array válido encontrado no .mat.")
            de_key = max(arrays, key=lambda k: arrays[k].size)
            print(f"[AVISO] Chave '_DE_time' não encontrada. Usando a maior matriz: {de_key}")
            
        return mat_data[de_key].flatten()

    def create_windows(self, time_series: np.ndarray, label: int) -> Tuple[np.ndarray, np.ndarray]:
        num_windows = (len(time_series) - self.window_size) // self.step_size + 1
        if num_windows <= 0:
             return np.array([]), np.array([])
        
        windows = np.array([
            time_series[i * self.step_size : i * self.step_size + self.window_size] 
            for i in range(num_windows)
        ])
        
        windows = np.expand_dims(windows, axis=-1)
        labels = np.full((num_windows,), label)
        
        return windows, labels
