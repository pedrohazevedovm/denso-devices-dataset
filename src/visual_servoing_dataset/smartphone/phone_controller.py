import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Any
from PIL import Image, ImageDraw

import cv2
import numpy as np
import requests
from requests.exceptions import RequestException

from ..config import SmartphoneConfig
from ..utils.logger import logger


def _create_simulated_image(
    target_path: Path,
    camera_name: str,
    filename: str,
    extra_text: Optional[str] = None
) -> None:
    """Gera uma imagem sintética 1920x1080 para uso no modo dry-run / simulação."""
    img = Image.new("RGB", (1920, 1080), color=(30, 34, 42))
    draw = ImageDraw.Draw(img)

    draw.rectangle([(20, 20), (1900, 1060)], outline=(0, 200, 255), width=4)
    draw.text((40, 40), f"Camera: {camera_name}", fill=(255, 255, 255))
    draw.text((40, 80), f"File: {filename}", fill=(0, 255, 128))
    draw.text((40, 120), f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", fill=(200, 200, 200))
    if extra_text:
        draw.text((40, 160), extra_text, fill=(255, 200, 80))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_path, "JPEG", quality=100)


def parse_capture_headers(headers: Mapping[str, str]) -> Dict[str, Any]:
    """Extrai o JSON completo do header X-Camera-Metadata e campos de conveniência do Cam-App."""
    metadata: Dict[str, Any] = {}
    raw_json_header = headers.get("X-Camera-Metadata")
    if raw_json_header:
        try:
            parsed = json.loads(raw_json_header)
            if isinstance(parsed, dict):
                metadata.update(parsed)
        except json.JSONDecodeError as exc:
            logger.warning(f"Falha ao decodificar header JSON X-Camera-Metadata: {exc}")

    exp_ns = metadata.get("sensor_exposure_time")
    if exp_ns is None and "X-Camera-Exposure-Time-Ns" in headers:
        try:
            exp_ns = int(headers["X-Camera-Exposure-Time-Ns"])
        except ValueError:
            exp_ns = headers["X-Camera-Exposure-Time-Ns"]

    exp_ms_header = headers.get("X-Camera-Exposure-Time-Ms")
    if isinstance(exp_ns, (int, float)):
        exp_ms_val: Any = round(float(exp_ns) / 1_000_000.0, 4)
    elif exp_ms_header is not None:
        try:
            exp_ms_val = float(exp_ms_header)
        except ValueError:
            exp_ms_val = exp_ms_header
    else:
        exp_ms_val = None

    focus_val = metadata.get("lens_focus_distance")
    if focus_val is None and "X-Camera-Focus-Distance" in headers:
        try:
            focus_val = float(headers["X-Camera-Focus-Distance"])
        except ValueError:
            focus_val = headers["X-Camera-Focus-Distance"]

    iso_val = metadata.get("sensor_sensitivity")
    if iso_val is None and "X-Camera-ISO" in headers:
        try:
            iso_val = int(headers["X-Camera-ISO"])
        except ValueError:
            iso_val = headers["X-Camera-ISO"]

    metadata.setdefault("focus_distance", focus_val)
    metadata.setdefault("iso", iso_val)
    metadata.setdefault("exposure_time_ns", exp_ns)
    metadata.setdefault("exposure_time_ms", exp_ms_val)

    if "sensor_timestamp" not in metadata and "X-Camera-Sensor-Timestamp" in headers:
        try:
            metadata["sensor_timestamp"] = int(headers["X-Camera-Sensor-Timestamp"])
        except ValueError:
            metadata["sensor_timestamp"] = headers["X-Camera-Sensor-Timestamp"]

    return metadata


def _build_simulated_metadata(params: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Constrói metadados simulados compatíveis com as chaves do Cam-App para modo dry-run."""
    is_auto = str(params.get("control_mode", "manual")).lower() == "auto"
    focus_d = 1.8621974 if is_auto else float(params.get("focus_distance", 1.8621974))
    iso_val = 1141 if is_auto else int(params.get("iso", 1141))
    exp_ns = 16666660 if is_auto else int(params.get("exposure_time_ns", 16666660))
    ois_on = bool(params.get("ois_enabled", is_auto))
    eis_on = bool(params.get("eis_enabled", is_auto))

    return {
        "preset": preset_name,
        "sensor_exposure_time": exp_ns,
        "sensor_sensitivity": iso_val,
        "sensor_timestamp": time.time_ns(),
        "sensor_rolling_shutter_skew": 14192000,
        "lens_focus_distance": focus_d,
        "lens_focal_length": 4.52,
        "lens_aperture": 1.7,
        "lens_state": 0,
        "lens_intrinsic_calibration": [1460.0, 1460.0, 960.0, 540.0, 0.0],
        "camera_matrix_k": [1460.0, 0.0, 960.0, 0.0, 1460.0, 540.0, 0.0, 0.0, 1.0],
        "camera_matrix_width": 1920,
        "camera_matrix_height": 1080,
        "control_ae_mode": 1 if is_auto else 0,
        "control_af_mode": 4 if is_auto else 0,
        "control_awb_mode": 1 if is_auto else 0,
        "color_correction_mode": 2 if is_auto else 0,
        "color_correction_gains": {
            "red": 1.8,
            "green_even": 1.0,
            "green_odd": 1.0,
            "blue": 1.6,
        },
        "color_correction_transform": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "lens_optical_stabilization_mode": 1 if ois_on else 0,
        "control_video_stabilization_mode": 1 if eis_on else 0,
        "noise_reduction_mode": 2 if is_auto else 0,
        "edge_mode": 2 if is_auto else 0,
        "color_correction_aberration_mode": 2 if is_auto else 0,
        "scaler_crop_region": {"left": 0, "top": 0, "right": 4096, "bottom": 2304},
        "jpeg_quality": 100,
        "focus_distance": focus_d,
        "iso": iso_val,
        "exposure_time_ns": exp_ns,
        "exposure_time_ms": round(exp_ns / 1_000_000.0, 4),
        "capture_latency_ms": 45.0,
    }


def _to_config_payload(params: Dict[str, Any]) -> Dict[str, Any]:
    """Filtra e formata o dicionário de preset para o JSON esperado pelo POST /config do Cam-App."""
    control_mode = str(params.get("control_mode", "manual")).lower()
    is_auto = control_mode == "auto"
    return {
        "control_mode": "auto" if is_auto else "manual",
        "auto_mode": is_auto,
        "focus_distance": float(params.get("focus_distance", 0.0)),
        "iso": int(params.get("iso", 100)),
        "exposure_time_ns": int(params.get("exposure_time_ns", 10000000)),
        "ois_enabled": bool(params.get("ois_enabled", is_auto)),
        "eis_enabled": bool(params.get("eis_enabled", is_auto)),
    }


class SmartphoneController:
    """Controlador HTTP para o aplicativo Android Cam-App (Camera2 HAL Server na porta 8080).

    Opera exclusivamente com duas configurações de câmera no Cam-App:
      - 'auto': Modo 3A (AF, AE, AWB) + filtros ISP + OIS/EIS ativados.
      - 'manual_calibrado': Exposição, sensibilidade, foco e balanço de branco fixos,
        com controles de pós-processamento e estabilização desativáveis configurados para OFF,
        calibrados a partir do modo 'auto' (ou fixados no config.yaml).
    """

    def __init__(self, config: SmartphoneConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "VisualServoingDatasetCollector/2.0",
            "Accept": "*/*",
        })
        self.connected_devices: List[str] = []
        self.device_names: Dict[str, str] = {}
        self.device_info: Dict[str, Dict[str, Any]] = {}
        self.device_presets: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.current_presets: Dict[str, str] = {}
        self.last_applied_settings: Dict[str, Dict[str, Any]] = {}
        self.last_telemetry: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Controlador Cam-App HTTP inicializado (configs ativas: [bold yellow]%s[/bold yellow])",
            ", ".join(self.config.active_presets),
        )

    @property
    def connected_ips(self) -> List[str]:
        return self.connected_devices

    def _get_base_url(self, device_id: str) -> str:
        if ":" in device_id:
            return f"http://{device_id}"
        return f"http://{device_id}:{self.config.port}"

    def connect(self) -> bool:
        """Testa conexão HTTP com todos os smartphones configurados via GET /status."""
        logger.info(f"--- [Cam-App] Conectando aos smartphones via HTTP (Porta {self.config.port}) ---")

        if self.dry_run:
            logger.info("[DRY-RUN] Simulação de conexão com Cam-App Android.")
            target_ips = self.config.ips if self.config.ips else ["192.168.155.44"]
            self.connected_devices = list(target_ips)
            self._resolve_device_names()
            for dev in self.connected_devices:
                self.device_presets[dev] = copy.deepcopy(self.config.presets)
                self.device_info[dev] = {
                    "status": "online",
                    "device_model": "Simulated Moto Razr 60 Ultra",
                    "android_version": "15",
                }
                name = self.device_names.get(dev, dev)
                logger.info(f"[DRY-RUN] Cam-App simulado: {name} (IP: {dev})")
            return True

        if not self.config.ips:
            logger.error("Nenhum IP de smartphone configurado em 'smartphone.ips'!")
            return False

        self.connected_devices.clear()
        for ip in self.config.ips:
            base_url = self._get_base_url(ip)
            status_url = f"{base_url}/status"
            try:
                logger.info(f"Verificando Cam-App em: {status_url}")
                resp = self.session.get(status_url, timeout=self.config.timeout_sec)
                if resp.status_code == 200:
                    data = resp.json()
                    model = data.get("device_model", "Android")
                    android_ver = data.get("android_version", "Desconhecido")
                    captures = data.get("captures_served", 0)
                    logger.info(
                        f"Cam-App online: {ip} | Modelo: {model} (Android {android_ver}) | Capturas prévias: {captures}"
                    )
                    self.connected_devices.append(ip)
                    self.device_info[ip] = data
                    self.device_presets[ip] = copy.deepcopy(self.config.presets)
                else:
                    logger.warning(f"Cam-App {ip} respondeu com status {resp.status_code}")
            except Exception as e:
                logger.warning(f"Não foi possível conectar ao Cam-App em {ip}: {e}")

        self._resolve_device_names()
        if self.connected_devices:
            logger.info(f"Total de dispositivos Cam-App conectados: {len(self.connected_devices)}")
            return True

        logger.error("Nenhum smartphone respondeu na rede. Verifique o IP e se o Cam-App está aberto.")
        return False

    def get_device_presets(self, device_id: str) -> Dict[str, Dict[str, Any]]:
        """Retorna os presets ('auto' e 'manual_calibrado') ativos para um dispositivo específico."""
        if device_id not in self.device_presets:
            self.device_presets[device_id] = copy.deepcopy(self.config.presets)
        return self.device_presets[device_id]

    def calibrate_manual_from_auto(
        self,
        device_id: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Calibra os parâmetros de 'manual_calibrado' com base na leitura real do modo 'auto'.

        Fluxo:
          1. Aplica o preset 'auto' no Cam-App e aguarda acomodação do 3A (AF/AE/AWB).
          2. Realiza uma captura de referência em GET /capture e extrai do X-Camera-Metadata:
             - lens_focus_distance (dioptrias)
             - sensor_sensitivity (ISO)
             - sensor_exposure_time (ns)
          3. Atualiza o preset 'manual_calibrado' do dispositivo com esses parâmetros e
             control_mode='manual', ois_enabled=False, eis_enabled=False.
        """
        targets = [device_id] if device_id else self.connected_devices
        calibrated_by_device: Dict[str, Dict[str, Any]] = {}

        for dev in targets:
            device_label = self.device_names.get(dev, dev)
            presets = self.get_device_presets(dev)
            manual_cfg = presets.get("manual_calibrado", {})
            should_auto_calibrate = force or bool(
                manual_cfg.get("auto_calibrate", self.config.auto_calibrate_manual)
            )

            if not should_auto_calibrate:
                logger.info(
                    f"[{device_label}] 'auto_calibrate' desativado para 'manual_calibrado'. "
                    f"Usando parâmetros fixos do config.yaml: "
                    f"Foco={manual_cfg.get('focus_distance')} D, "
                    f"ISO={manual_cfg.get('iso')}, "
                    f"Exp={manual_cfg.get('exposure_time_ns')} ns"
                )
                calibrated_by_device[dev] = copy.deepcopy(manual_cfg)
                continue

            logger.info(
                f"[{device_label}] Calibrando 'manual_calibrado' a partir do modo 'auto'..."
            )
            if not self.apply_preset("auto", device_id=dev):
                logger.warning(
                    f"[{device_label}] Falha ao aplicar modo 'auto' para calibração. "
                    f"Mantendo valores padrão de 'manual_calibrado'."
                )
                calibrated_by_device[dev] = copy.deepcopy(manual_cfg)
                continue

            if self.dry_run:
                sim_meta = _build_simulated_metadata(presets["auto"], "auto")
                focus_d = float(sim_meta["lens_focus_distance"])
                iso_val = int(sim_meta["sensor_sensitivity"])
                exp_ns = int(sim_meta["sensor_exposure_time"])
            else:
                _, raw_bytes, meta = self._capture_raw(dev)
                if not raw_bytes or not meta:
                    logger.warning(
                        f"[{device_label}] Falha ao capturar frame de referência no modo 'auto'. "
                        f"Mantendo valores de fallback em 'manual_calibrado'."
                    )
                    calibrated_by_device[dev] = copy.deepcopy(manual_cfg)
                    continue

                focus_d = float(
                    meta.get("lens_focus_distance", meta.get("focus_distance", manual_cfg.get("focus_distance", 0.0)))
                )
                iso_val = int(
                    meta.get("sensor_sensitivity", meta.get("iso", manual_cfg.get("iso", 100)))
                )
                exp_ns = int(
                    meta.get("sensor_exposure_time", meta.get("exposure_time_ns", manual_cfg.get("exposure_time_ns", 10000000)))
                )

            presets["manual_calibrado"].update({
                "control_mode": "manual",
                "focus_distance": focus_d,
                "iso": iso_val,
                "exposure_time_ns": exp_ns,
                "ois_enabled": False,
                "eis_enabled": False,
                "calibrated_from_auto": True,
            })
            calibrated_by_device[dev] = copy.deepcopy(presets["manual_calibrado"])
            logger.info(
                f"[{device_label}] 'manual_calibrado' calibrado pelo modo 'auto': "
                f"Foco={focus_d:.4f} D | ISO={iso_val} | Exposição={exp_ns} ns ({exp_ns / 1e6:.2f} ms)"
            )

        return calibrated_by_device

    def prepare_camera(self, device_id: str, preset_name: Optional[str] = None) -> bool:
        """Prepara a câmera aplicando um preset inicial caso especificado."""
        device_label = self.device_names.get(device_id, device_id)
        logger.info(f"Preparando Cam-App em {device_label}...")
        if preset_name:
            return self.apply_preset(preset_name, device_id=device_id)
        return True

    def apply_preset(self, preset_name: str, device_id: Optional[str] = None) -> bool:
        """Envia os parâmetros do preset ('auto' ou 'manual_calibrado') via POST /config."""
        targets = [device_id] if device_id else self.connected_devices
        success = True

        for dev in targets:
            device_label = self.device_names.get(dev, dev)
            presets = self.get_device_presets(dev)
            if preset_name not in presets:
                logger.error(
                    f"Configuração de câmera '{preset_name}' não encontrada em {device_label}. "
                    f"Opções válidas: {list(presets.keys())}"
                )
                success = False
                continue

            raw_params = presets[preset_name]
            payload = _to_config_payload(raw_params)
            logger.info(f"Aplicando configuração '{preset_name}' em {device_label}: {payload}")

            if self.dry_run:
                self.current_presets[dev] = preset_name
                self.last_applied_settings[dev] = dict(payload)
                self.last_telemetry[dev] = _build_simulated_metadata(payload, preset_name)
                continue

            if self._post_config_with_retry(dev, payload):
                self.current_presets[dev] = preset_name
                logger.info(
                    f"Configuração '{preset_name}' aplicada em {device_label} "
                    f"(applied_settings={self.last_applied_settings.get(dev)})"
                )
            else:
                success = False

        settle_sec = (
            self.config.auto_3a_settle_sec
            if preset_name == "auto"
            else self.config.manual_settle_sec
        )
        if not self.dry_run and settle_sec > 0:
            logger.info(
                f"Aguardando {settle_sec:.2f}s para acomodação ({preset_name}) da lente (VCM) / sensor..."
            )
            time.sleep(settle_sec)

        return success

    def configure_camera(self, device_id: str, params: Dict[str, Any]) -> bool:
        """Envia um dicionário customizado de parâmetros para POST /config."""
        payload = _to_config_payload(params)
        if self.dry_run:
            self.last_applied_settings[device_id] = dict(payload)
            self.last_telemetry[device_id] = _build_simulated_metadata(payload, "custom")
            return True

        ok = self._post_config_with_retry(device_id, payload)
        if ok and self.config.stabilization_delay_sec > 0:
            time.sleep(self.config.stabilization_delay_sec)
        return ok

    def _post_config_with_retry(self, device_id: str, payload: Dict[str, Any]) -> bool:
        device_label = self.device_names.get(device_id, device_id)
        config_url = f"{self._get_base_url(device_id)}/config"

        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.post(config_url, json=payload, timeout=self.config.timeout_sec)
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                        self.last_applied_settings[device_id] = body.get("applied_settings", dict(payload))
                    except ValueError:
                        self.last_applied_settings[device_id] = dict(payload)
                    return True

                logger.warning(
                    f"[{device_label}] Tentativa {attempt}/{self.config.max_retries}: "
                    f"POST /config retornou status {resp.status_code} ({resp.text})"
                )
            except RequestException as exc:
                logger.warning(
                    f"[{device_label}] Tentativa {attempt}/{self.config.max_retries} falhou em POST /config: {exc}"
                )

            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_backoff_sec * (2 ** (attempt - 1)))

        logger.error(f"[{device_label}] Esgotadas as {self.config.max_retries} tentativas em POST /config.")
        return False

    def _capture_raw(self, device_id: str) -> Tuple[Optional[np.ndarray], Optional[bytes], Dict[str, Any]]:
        """Executa GET /capture com retentativas e extrai bytes JPEG e X-Camera-Metadata completo."""
        device_label = self.device_names.get(device_id, device_id)
        capture_url = f"{self._get_base_url(device_id)}/capture"

        for attempt in range(1, self.config.max_retries + 1):
            try:
                start_t = time.perf_counter()
                resp = self.session.get(capture_url, timeout=self.config.timeout_sec)
                elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

                if resp.status_code == 200 and resp.content:
                    raw_jpeg_bytes = bytes(resp.content)
                    img_array = np.frombuffer(raw_jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if frame is not None and frame.size > 0:
                        metadata = parse_capture_headers(resp.headers)
                        metadata["capture_latency_ms"] = elapsed_ms
                        return frame, raw_jpeg_bytes, metadata

                    logger.warning(f"[{device_label}] Tentativa {attempt}: Frame JPEG decodificado vazio ou corrompido.")
                else:
                    logger.warning(
                        f"[{device_label}] Tentativa {attempt}: GET /capture retornou status {resp.status_code}."
                    )
            except RequestException as exc:
                logger.warning(
                    f"[{device_label}] Tentativa {attempt}/{self.config.max_retries} erro de rede em GET /capture: {exc}"
                )

            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_backoff_sec * (2 ** (attempt - 1)))

        logger.error(f"[{device_label}] Falha ao capturar frame após {self.config.max_retries} tentativas.")
        return None, None, {}

    def capture_and_save_photo(
        self,
        device_id: str,
        target_dir: Path,
        target_filename: str
    ) -> Optional[Path]:
        """Captura foto via GET /capture no Cam-App, extrai X-Camera-Metadata e salva os bytes JPEG."""
        device_label = self.device_names.get(device_id, device_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / target_filename
        active_preset = self.current_presets.get(device_id, "auto")

        logger.info(
            f"[Cam-App] Capturando foto ({active_preset}) com {device_label} -> {final_path}"
        )

        if self.dry_run:
            presets = self.get_device_presets(device_id)
            params = presets.get(active_preset, {})
            sim_meta = _build_simulated_metadata(params, active_preset)
            self.last_telemetry[device_id] = sim_meta
            extra = (
                f"Preset: {active_preset} | Mode: {params.get('control_mode', 'auto')} | "
                f"Focus: {sim_meta['lens_focus_distance']:.2f}D | ISO: {sim_meta['sensor_sensitivity']} | "
                f"Exp: {sim_meta['exposure_time_ms']}ms"
            )
            _create_simulated_image(final_path, device_label, target_filename, extra_text=extra)
            return final_path

        frame, raw_jpeg_bytes, metadata = self._capture_raw(device_id)
        if frame is not None and raw_jpeg_bytes is not None:
            metadata["preset"] = active_preset
            self.last_telemetry[device_id] = metadata

            if metadata.get("lens_state") == 1:
                logger.warning(
                    f"[{device_label}] Atenção: Frame {target_filename} capturado com lente em movimento (lens_state=1)!"
                )

            final_path.write_bytes(raw_jpeg_bytes)
            logger.info(
                f"Foto salva ({len(raw_jpeg_bytes)} bytes) | Preset: {active_preset} | "
                f"Foco: {metadata.get('lens_focus_distance')} D | ISO: {metadata.get('sensor_sensitivity')} | "
                f"Exp: {metadata.get('exposure_time_ms')} ms | Latência: {metadata.get('capture_latency_ms')} ms"
            )
            return final_path

        return None

    def get_last_telemetry(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Retorna o dicionário completo de metadados (X-Camera-Metadata) da última captura."""
        return self.last_telemetry.get(device_id)

    def get_last_applied_settings(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Retorna o applied_settings retornado pelo Cam-App no último POST /config."""
        return self.last_applied_settings.get(device_id)

    def close_cameras(self) -> None:
        logger.info("Encerrando sessão HTTP com o Cam-App...")
        try:
            self.session.close()
        except Exception:
            pass

    def _resolve_device_names(self) -> None:
        self.device_names.clear()
        for idx, dev in enumerate(self.connected_devices, start=1):
            if dev in self.config.camera_names:
                name = self.config.camera_names[dev]
            else:
                name = f"cam_{idx:02d}_{dev.replace('.', '_').replace(':', '_')}"
            self.device_names[dev] = name
