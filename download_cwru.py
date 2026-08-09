import os
import requests

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Baixando {url} para {target_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download concluído.")
    else:
        print(f"Arquivo já existe: {target_path}")

def main():
    print("Baixando datasets complementares oficiais do CWRU (Case Western Reserve University)...")
    base_url = "https://engineering.case.edu/sites/default/files/"
    
    # Inner Race fault (Falha de Pista Interna) - Arquivo 105.mat
    download_file(base_url + "105.mat", "data/1_inner_race/105.mat")
    
    # Outer Race fault (Falha de Pista Externa) - Arquivo 130.mat
    download_file(base_url + "130.mat", "data/2_outer_race/130.mat")
    
    print("Pronto! Agora temos dados de diferentes falhas para cruzar com a sua classe normal.")

if __name__ == "__main__":
    main()
