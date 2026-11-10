import json
import os
import time
from datetime import datetime
from typing import Any, Dict

import requests

# Endpoints do aplicativo Cam-App Android (porta 8080)
BASE_URL = os.getenv("CAM_BASE_URL", "http://192.168.155.44:8080")
CAPTURE_URL = f"{BASE_URL}/capture"
CONFIG_URL = f"{BASE_URL}/config"
STATUS_URL = f"{BASE_URL}/status"

# Apenas as duas configurações utilizadas: auto e manual_calibrado (baseado no auto)
CONFIGS: Dict[str, Dict[str, Any]] = {
    "auto": {
        "control_mode": "auto",
        "focus_distance": 0.0,
        "iso": 100,
        "exposure_time_ns": 10000000,
        "ois_enabled": True,
        "eis_enabled": True,
    },
    "manual_calibrado": {
        "control_mode": "manual",
        "focus_distance": 1.8621974,
        "iso": 1141,
        "exposure_time_ns": 16666660,
        "ois_enabled": False,
        "eis_enabled": False,
    },
}

SAMPLES_PER_CONFIG = 3
VCM_SETTLING_DELAY_S = 1.0
BASE_DIR = "dataset_servovisao"


def parse_metadata(headers: Any) -> Dict[str, Any]:
    raw_meta = headers.get("X-Camera-Metadata", "{}")
    try:
        meta = json.loads(raw_meta)
    except json.JSONDecodeError:
        meta = {}
    return meta


def calibrate_manual_from_auto(session: requests.Session) -> None:
    """Coloca o Cam-App em modo 'auto', captura 1 frame e atualiza 'manual_calibrado'."""
    print("\n--- Calibrando 'manual_calibrado' a partir do modo 'auto' ---")
    resp = session.post(CONFIG_URL, json=CONFIGS["auto"], timeout=3.0)
    resp.raise_for_status()
    time.sleep(VCM_SETTLING_DELAY_S)

    cap_resp = session.get(CAPTURE_URL, timeout=3.0)
    cap_resp.raise_for_status()
    meta = parse_metadata(cap_resp.headers)

    focus_d = float(meta.get("lens_focus_distance", CONFIGS["manual_calibrado"]["focus_distance"]))
    iso_val = int(meta.get("sensor_sensitivity", CONFIGS["manual_calibrado"]["iso"]))
    exp_ns = int(meta.get("sensor_exposure_time", CONFIGS["manual_calibrado"]["exposure_time_ns"]))

    CONFIGS["manual_calibrado"].update({
        "focus_distance": focus_d,
        "iso": iso_val,
        "exposure_time_ns": exp_ns,
    })
    print(
        f"Parâmetros calibrados do modo auto -> "
        f"focus_distance={focus_d:.4f} D, iso={iso_val}, exposure_time_ns={exp_ns}"
    )


def collect_dataset() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Iniciando coleta com Cam-App ({BASE_URL}). Saída: {BASE_DIR}/")

    with requests.Session() as session:
        try:
            calibrate_manual_from_auto(session)
        except requests.exceptions.RequestException as e:
            print(f"Aviso: falha ao calibrar automaticamente do modo auto ({e}). Usando valores padrão.")

        for config_name, params in CONFIGS.items():
            print(f"\n--- Aplicando configuração: {config_name} ---")
            try:
                config_resp = session.post(CONFIG_URL, json=params, timeout=3.0)
                if config_resp.status_code != 200:
                    print(f"Erro ao configurar {config_name}. Status: {config_resp.status_code}")
                    continue
                applied = config_resp.json().get("applied_settings", params)
                print(f"Configuração aplicada: {applied}")
            except requests.exceptions.RequestException as e:
                print(f"Erro de conexão ao enviar configuração: {e}")
                continue

            time.sleep(VCM_SETTLING_DELAY_S)

            config_dir = os.path.join(BASE_DIR, config_name)
            os.makedirs(config_dir, exist_ok=True)
            records = []

            for i in range(SAMPLES_PER_CONFIG):
                try:
                    print(f"Capturando imagem {i + 1}/{SAMPLES_PER_CONFIG}...")
                    response = session.get(CAPTURE_URL, timeout=3.0)

                    if response.status_code == 200 and response.content:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        filename = f"img_{timestamp}.jpg"
                        filepath = os.path.join(config_dir, filename)

                        with open(filepath, "wb") as f:
                            f.write(response.content)

                        meta = parse_metadata(response.headers)
                        records.append({"filename": filename, "metadata": meta})
                        print(
                            f"Salvo: {filepath} | "
                            f"Foco={meta.get('lens_focus_distance')} D, "
                            f"ISO={meta.get('sensor_sensitivity')}, "
                            f"Exp={meta.get('sensor_exposure_time')} ns"
                        )
                        time.sleep(0.2)
                    else:
                        print(f"Falha na captura. Status: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    print(f"Erro de conexão na captura: {e}")

            meta_path = os.path.join(config_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(
                    {"scenario_name": config_name, "requested_config": params, "frames": records},
                    mf,
                    indent=2,
                    ensure_ascii=False,
                )

    print("\nColeta finalizada com sucesso!")


if __name__ == "__main__":
    collect_dataset()
