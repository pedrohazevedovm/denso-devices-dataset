import cv2
import numpy as np
import requests
import time
import os
from datetime import datetime

# Endpoints do app Android
CAMERA_URL = "http://192.168.158.48:8080/capture"
CONFIG_URL = "http://192.168.158.48:8080/config"

# Os 4 cenários de teste definidos
CONFIGS = {
    "1_home_visao_geral": {
        "focus_distance": 1.66,
        "iso": 100,
        "exposure_time_ns": 8333333
    },
    "2_aproximacao_fina": {
        "focus_distance": 5.0,
        "iso": 100,
        "exposure_time_ns": 10000000
    },
    "3_camera_em_movimento": {
        "focus_distance": 2.5,
        "iso": 400,
        "exposure_time_ns": 2000000
    },
    "4_baixa_luminosidade": {
        "focus_distance": 2.5,
        "iso": 800,
        "exposure_time_ns": 16666666
    }
}

# Parâmetros de coleta
SAMPLES_PER_CONFIG = 3  # Quantidade de fotos por configuração
BASE_DIR = "dataset_servovisao"


def collect_dataset():
    # Cria o diretório base se não existir
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Iniciando coleta. As imagens serão salvas em: {BASE_DIR}/")

    for config_name, params in CONFIGS.items():
        print(f"\n--- Aplicando configuração: {config_name} ---")

        # 1. Envia a configuração para a câmera
        try:
            config_resp = requests.post(CONFIG_URL, json=params, timeout=2.0)
            if config_resp.status_code != 200:
                print(f"Erro ao configurar {config_name}. Status: {config_resp.status_code}")
                continue

            print(f"Configuração aplicada: {params}")
        except requests.exceptions.RequestException as e:
            print(f"Erro de conexão ao enviar configuração: {e}")
            continue

        # 2. Aguarda o sensor de hardware aplicar e estabilizar os ganhos analógicos/foco
        time.sleep(0.5)

        # Cria a subpasta para a configuração atual
        config_dir = os.path.join(BASE_DIR, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # 3. Loop de captura para a configuração atual
        for i in range(SAMPLES_PER_CONFIG):
            try:
                print(f"Capturando imagem {i + 1}/{SAMPLES_PER_CONFIG}...")
                response = requests.get(CAMERA_URL, timeout=2.0)

                if response.status_code == 200:
                    # Decodifica o buffer JPEG diretamente para um array BGR do OpenCV
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    # Gera um nome de arquivo com timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"img_{timestamp}.jpg"
                    filepath = os.path.join(config_dir, filename)

                    # Salva a imagem sem compressão adicional que prejudique a qualidade
                    cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    print(f"Salvo: {filepath}")

                    # Opcional: Mostra a imagem rapidamente (comentado para rodar headless)
                    cv2.imshow("Robot Arm View", frame)
                    cv2.waitKey(0) # Mostra por 200ms

                    # Pausa entre capturas da mesma configuração
                    time.sleep(0.2)
                else:
                    print(f"Falha na captura. Status: {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"Erro de conexão na captura: {e}")

    # cv2.destroyAllWindows()
    print("\nColeta finalizada com sucesso!")


if __name__ == "__main__":
    collect_dataset()