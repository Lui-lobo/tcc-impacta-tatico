import numpy as np
import scipy.io
from scipy import signal
from typing import List, Tuple


class DataProcessor:
    """Carrega, reamostra e fatia as series temporais do dataset CWRU.

    A reamostragem existe porque o hardware de destino nao alcanca a taxa
    original do dataset: o acelerometro do MPU6050 atualiza os registradores a
    no maximo 1 kHz, enquanto o CWRU foi gravado a 12 kHz (Drive-End) ou 48 kHz
    (Normal Baseline). Treinar na taxa original e inferir a 1 kHz colocaria o
    modelo diante de um conteudo espectral que ele nunca viu.
    """

    def __init__(self, window_size: int = 512, overlap: float = 0.5):
        self.window_size = window_size
        self.step_size = max(1, int(window_size * (1 - overlap)))

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

    @staticmethod
    def decimation_stages(factor: int, max_stage: int = 8) -> List[int]:
        """Fatora o fator de decimacao em etapas pequenas.

        `scipy.signal.decimate` perde qualidade com fatores grandes porque o
        filtro anti-aliasing precisa de uma transicao cada vez mais estreita.
        Decimar 48x de uma vez introduz ondulacao na banda passante; decimar
        8x e depois 6x mantem cada filtro numa regiao numericamente saudavel.
        """
        if factor < 1:
            raise ValueError(f"Fator de decimação inválido: {factor}")
        stages: List[int] = []
        remaining = factor
        while remaining > max_stage:
            for q in range(max_stage, 1, -1):
                if remaining % q == 0:
                    stages.append(q)
                    remaining //= q
                    break
            else:
                raise ValueError(
                    f"Fator {factor} tem um fator primo maior que {max_stage}; "
                    "escolha taxas de origem/destino com razão mais amigável."
                )
        if remaining > 1:
            stages.append(remaining)
        return stages

    def resample_to(self, time_series: np.ndarray, source_rate: int,
                    target_rate: int) -> np.ndarray:
        """Reduz a taxa de amostragem aplicando filtro anti-aliasing.

        Usa FIR com `zero_phase=True`: filtra para frente e para tras, o que
        cancela o atraso de fase. Preservar a fase importa porque as assinaturas
        de falha em rolamento sao impulsos periodicos, e um filtro de fase nao
        linear distorceria justamente o formato desses impulsos.
        """
        if source_rate == target_rate:
            return time_series.astype(np.float64)
        if source_rate < target_rate:
            raise ValueError(
                f"Não é possível aumentar a taxa ({source_rate} Hz -> {target_rate} Hz)."
            )
        if source_rate % target_rate != 0:
            raise ValueError(
                f"Taxa de origem {source_rate} Hz não é múltipla inteira de "
                f"{target_rate} Hz."
            )

        out = time_series.astype(np.float64)
        for q in self.decimation_stages(source_rate // target_rate):
            out = signal.decimate(out, q, ftype='fir', zero_phase=True)
        return out

    def split_time_series(self, time_series: np.ndarray, test_ratio: float = 0.2
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Separa treino e teste em trechos CONTIGUOS, antes do janelamento.

        Janelas com sobreposicao compartilham amostras. Se o corte treino/teste
        fosse feito depois do janelamento (train_test_split aleatorio), uma
        janela de teste teria ate 90% das amostras em comum com uma janela de
        treino, e a acuracia medida seria fruto de vazamento de dados.
        Cortando a serie antes, e descartando uma janela inteira na fronteira,
        nenhuma amostra aparece dos dois lados.
        """
        n = len(time_series)
        cut = int(n * (1.0 - test_ratio))
        # A zona morta de `window_size` amostras sai do lado do TREINO. Descontar
        # do teste encolheria demais o conjunto de avaliacao - no arquivo normal,
        # que e o mais curto, sobrariam menos amostras que uma janela inteira.
        train_end = max(0, cut - self.window_size)
        return time_series[:train_end], time_series[cut:]

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
