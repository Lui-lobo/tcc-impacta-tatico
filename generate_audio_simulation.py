import os
import scipy.io
from scipy.io import wavfile
import numpy as np

def mat_to_wav(mat_filepath, wav_filepath, sample_rate=12000):
    """
    Converts a CWRU .mat vibration file into a playable .wav audio file.
    """
    print(f"Lendo {mat_filepath}...")
    
    # 1. Load the .mat file
    try:
        mat_data = scipy.io.loadmat(mat_filepath)
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    # 2. Extract the Drive End (DE) accelerometer data
    de_key = next((key for key in mat_data.keys() if '_DE_time' in key), None)
    
    if not de_key:
        print("Chave '_DE_time' não encontrada. Tentando usar a maior matriz...")
        arrays = {k: v for k, v in mat_data.items() if not k.startswith('__') and isinstance(v, np.ndarray)}
        if not arrays:
            print("Nenhum array válido encontrado.")
            return
        de_key = max(arrays, key=lambda k: arrays[k].size)

    # Flatten the array to a 1D sequence
    vibration_data = mat_data[de_key].flatten()

    # 3. Normalize the data to standard 16-bit PCM Audio (-32768 to 32767)
    # The vibration data is usually in float (g-force). Audio needs to be integers.
    max_val = np.max(np.abs(vibration_data))
    
    if max_val == 0:
        print("Aviso: Dados vazios ou zerados.")
        return
        
    # Scale to 16-bit range
    normalized_data = (vibration_data / max_val) * 32767
    
    # Convert to 16-bit integers
    audio_data = np.int16(normalized_data)

    # 4. Save as .wav file
    # Ensure the directory exists
    os.makedirs(os.path.dirname(wav_filepath), exist_ok=True)
    
    wavfile.write(wav_filepath, sample_rate, audio_data)
    print(f"Sucesso! Áudio salvo em: {wav_filepath}")

def main():
    print("=== Gerador de Simulação por Áudio (CWRU para WAV) ===")
    
    # Ensure the data exists
    if not os.path.exists("data/1_inner_race/105.mat"):
        print("Arquivos .mat não encontrados! Rode o 'python download_cwru.py' primeiro.")
        return

    # Create output directory
    os.makedirs("data/audio_simulation", exist_ok=True)

    # Convert Normal Baseline (Healthy)
    if os.path.exists("data/0_normal/97.mat"):
        mat_to_wav(
            mat_filepath="data/0_normal/97.mat", 
            wav_filepath="data/audio_simulation/normal_baseline.wav"
        )
    else:
        print("Arquivo normal (97.mat) não encontrado.")

    # Convert Inner Race fault
    mat_to_wav(
        mat_filepath="data/1_inner_race/105.mat", 
        wav_filepath="data/audio_simulation/inner_race_fault.wav"
    )
    
    # Convert Outer Race fault
    mat_to_wav(
        mat_filepath="data/2_outer_race/130.mat", 
        wav_filepath="data/audio_simulation/outer_race_fault.wav"
    )
    
    print("\nProcesso concluído! Agora você pode reproduzir esses arquivos .wav no seu alto-falante.")

if __name__ == "__main__":
    main()
