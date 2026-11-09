import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageDraw

import requests

from ..config import SmartphoneConfig
from ..utils.logger import logger

try:
    from device_manager import DeviceManager
    DEVICE_MANAGER_AVAILABLE = True
except ImportError:
    DEVICE_MANAGER_AVAILABLE = False
    DeviceManager = None


def _create_simulated_image(
    target_path: Path,
    camera_name: str,
    filename: str,
    extra_text: Optional[str] = None
) -> None:
    """Gera uma imagem sintética para uso no modo dry-run / simulação."""
    img = Image.new("RGB", (640, 480), color=(30, 34, 42))
    draw = ImageDraw.Draw(img)

    # Borda visual demonstrativa
    draw.rectangle([(10, 10), (630, 470)], outline=(0, 200, 255), width=3)
    draw.text((30, 30), f"Camera: {camera_name}", fill=(255, 255, 255))
    draw.text((30, 60), f"File: {filename}", fill=(0, 255, 128))
    draw.text((30, 90), f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", fill=(200, 200, 200))
    if extra_text:
        draw.text((30, 120), extra_text, fill=(255, 200, 80))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_path, "JPEG")


class IPCameraBackend:
    """Backend para o aplicativo headless Android IP Camera (Camera2 HAL via HTTP)."""

    def __init__(self, config: SmartphoneConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.session = requests.Session()
        self.connected_devices: List[str] = []
        self.device_names: Dict[str, str] = {}
        self.last_telemetry: Dict[str, Dict[str, Any]] = {}
        self.current_presets: Dict[str, str] = {}

    def _get_base_url(self, device_id: str) -> str:
        if ":" in device_id:
            return f"http://{device_id}"
        return f"http://{device_id}:{self.config.port}"

    def connect(self) -> bool:
        logger.info(f"--- [IP Camera] Conectando aos dispositivos via HTTP (Porta {self.config.port}) ---")

        if self.dry_run:
            logger.info("[DRY-RUN] Simulação de conexão com IP Camera Android.")
            target_ips = self.config.ips if self.config.ips else ["192.168.158.48"]
            self.connected_devices = target_ips
            self._resolve_device_names()
            for dev in self.connected_devices:
                name = self.device_names.get(dev, dev)
                logger.info(f"[DRY-RUN] Câmera IP simulada: {name} (IP: {dev})")
            return True

        if not self.config.ips:
            logger.error("Nenhum IP de smartphone configurado em 'smartphone.ips'!")
            return False

        self.connected_devices.clear()
        for ip in self.config.ips:
            base_url = self._get_base_url(ip)
            status_url = f"{base_url}/status"
            try:
                logger.info(f"Testando conexão com Câmera IP: {status_url}")
                resp = self.session.get(status_url, timeout=self.config.timeout_sec)
                if resp.status_code == 200:
                    data = resp.json()
                    model = data.get("device_model", "Android")
                    android_ver = data.get("android_version", "Desconhecido")
                    captures = data.get("captures_served", 0)
                    logger.info(
                        f"Câmera IP online: {ip} | Modelo: {model} (Android {android_ver}) | Capturas prévias: {captures}"
                    )
                    self.connected_devices.append(ip)
                else:
                    logger.warning(f"Câmera IP {ip} respondeu com status {resp.status_code}")
            except Exception as e:
                logger.warning(f"Não foi possível conectar à Câmera IP {ip}: {e}")

        self._resolve_device_names()
        if self.connected_devices:
            logger.info(f"Total de Câmeras IP conectadas com sucesso: {len(self.connected_devices)}")
            return True

        logger.error("Nenhuma Câmera IP respondeu na rede. Verifique o Wi-Fi e se o aplicativo está aberto.")
        return False

    def prepare_camera(self, device_id: str, preset_name: Optional[str] = None) -> bool:
        device_label = self.device_names.get(device_id, device_id)
        target_preset = preset_name or self.config.default_preset

        logger.info(f"Preparando Câmera IP {device_label}...")
        if target_preset:
            return self.apply_preset(target_preset, device_id=device_id)
        return True

    def apply_preset(self, preset_name: str, device_id: Optional[str] = None) -> bool:
        targets = [device_id] if device_id else self.connected_devices
        if preset_name not in self.config.presets:
            logger.warning(f"Preset '{preset_name}' não encontrado nas configurações. Mantendo parâmetros atuais.")
            return False

        params = self.config.presets[preset_name]
        success = True

        for dev in targets:
            device_label = self.device_names.get(dev, dev)
            logger.info(f"Aplicando preset '{preset_name}' em {device_label}: {params}")

            if self.dry_run:
                self.current_presets[dev] = preset_name
                self.last_telemetry[dev] = {
                    "preset": preset_name,
                    "focus_distance_diopters": params.get("focus_distance", 2.0),
                    "iso": params.get("iso", 100),
                    "exposure_time_ns": params.get("exposure_time_ns", 10000000),
                    "exposure_time_ms": params.get("exposure_time_ns", 10000000) / 1e6
                }
                time.sleep(0.1)
                continue

            config_url = f"{self._get_base_url(dev)}/config"
            try:
                resp = self.session.post(config_url, json=params, timeout=self.config.timeout_sec)
                if resp.status_code == 200:
                    self.current_presets[dev] = preset_name
                    logger.info(f"Preset '{preset_name}' aplicado com sucesso em {device_label}.")
                else:
                    logger.error(f"Falha ao aplicar preset em {device_label}. Status: {resp.status_code}")
                    success = False
            except Exception as e:
                logger.error(f"Erro de rede ao enviar preset para {device_label}: {e}")
                success = False

        if self.config.stabilization_delay_sec > 0:
            time.sleep(self.config.stabilization_delay_sec)

        return success

    def configure_camera(self, device_id: str, params: Dict[str, Any]) -> bool:
        device_label = self.device_names.get(device_id, device_id)
        logger.info(f"Configurando parâmetros manuais em {device_label}: {params}")

        if self.dry_run:
            self.last_telemetry[device_id] = {
                "focus_distance_diopters": params.get("focus_distance", 2.0),
                "iso": params.get("iso", 100),
                "exposure_time_ns": params.get("exposure_time_ns", 10000000),
                "exposure_time_ms": params.get("exposure_time_ns", 10000000) / 1e6
            }
            return True

        config_url = f"{self._get_base_url(device_id)}/config"
        try:
            resp = self.session.post(config_url, json=params, timeout=self.config.timeout_sec)
            if resp.status_code == 200:
                if self.config.stabilization_delay_sec > 0:
                    time.sleep(self.config.stabilization_delay_sec)
                return True
            logger.error(f"Erro ao configurar {device_label}. Status: {resp.status_code}")
            return False
        except Exception as e:
            logger.error(f"Erro ao comunicar com {device_label}: {e}")
            return False

    def capture_and_save_photo(
        self,
        device_id: str,
        target_dir: Path,
        target_filename: str
    ) -> Optional[Path]:
        device_label = self.device_names.get(device_id, device_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / target_filename

        logger.info(f"[IP Camera] Capturando foto com {device_label} -> {final_path}")

        if self.dry_run:
            preset = self.current_presets.get(device_id, self.config.default_preset or "default")
            extra = f"Preset: {preset} (Simulated Camera2 HTTP)"
            _create_simulated_image(final_path, device_label, target_filename, extra_text=extra)
            time.sleep(0.1)
            return final_path

        capture_url = f"{self._get_base_url(device_id)}/capture"
        try:
            resp = self.session.get(capture_url, timeout=self.config.timeout_sec)
            if resp.status_code == 200:
                # Extrair telemetria dos headers HTTP retornados pelo app
                focus_dist = resp.headers.get("X-Camera-Focus-Distance")
                iso = resp.headers.get("X-Camera-ISO")
                exp_ns = resp.headers.get("X-Camera-Exposure-Time-Ns")
                exp_ms = resp.headers.get("X-Camera-Exposure-Time-Ms")

                telemetry: Dict[str, Any] = {}
                if focus_dist is not None:
                    telemetry["focus_distance_diopters"] = float(focus_dist)
                if iso is not None:
                    telemetry["iso"] = int(iso)
                if exp_ns is not None:
                    telemetry["exposure_time_ns"] = int(exp_ns)
                if exp_ms is not None:
                    telemetry["exposure_time_ms"] = float(exp_ms)

                self.last_telemetry[device_id] = telemetry

                # Gravar diretamente os bytes JPEG originais
                final_path.write_bytes(resp.content)
                logger.info(f"Foto salva ({len(resp.content)} bytes). Telemetria: {telemetry}")
                return final_path
            else:
                logger.error(f"Falha na captura em {device_label}. Status: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Erro ao capturar foto em {device_label}: {e}")
            return None

    def get_last_telemetry(self, device_id: str) -> Optional[Dict[str, Any]]:
        return self.last_telemetry.get(device_id)

    def close_cameras(self) -> None:
        logger.info("Encerrando sessão HTTP das Câmeras IP...")
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


class ADBCameraBackend:
    """Backend legado para o aplicativo nativo de câmera via ADB (device_manager)."""

    def __init__(self, config: SmartphoneConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run or not DEVICE_MANAGER_AVAILABLE
        self.device_manager: Optional[Any] = None
        self.connected_devices: List[str] = []
        self.device_names: Dict[str, str] = {}

    def connect(self) -> bool:
        logger.info("--- [App Padrão ADB] Conectando aos smartphones Android via ADB ---")

        if self.dry_run:
            logger.info("[DRY-RUN] Simulação de conexão com smartphones Android (modo App Padrão).")
            if self.config.serials and "auto" not in self.config.serials:
                self.connected_devices = self.config.serials
            else:
                self.connected_devices = ["sim_serial_01", "sim_serial_02"]

            self._resolve_device_names()
            for serial, name in self.device_names.items():
                logger.info(f"[DRY-RUN] Smartphone simulado: {name} (Serial: {serial})")
            return True

        if not DEVICE_MANAGER_AVAILABLE:
            logger.error("A biblioteca 'device_manager' não está disponível para o modo 'default_app'!")
            return False

        try:
            logger.info("Inicializando DeviceManager e carregando dispositivos Android...")
            self.device_manager = DeviceManager()

            visible_devices = self.device_manager.connector.visible_devices()
            target_serials = []

            if self.config.serials and "auto" not in self.config.serials:
                for dev in visible_devices:
                    if dev.serial_number in self.config.serials or getattr(dev, "ip", None) in self.config.serials:
                        target_serials.append(dev.serial_number)
                if not target_serials:
                    target_serials = self.config.serials
            else:
                target_serials = [dev.serial_number for dev in visible_devices]

            if target_serials:
                self.connected_devices = self.device_manager.connect_devices(*target_serials)
            else:
                self.connected_devices = self.device_manager.connected_devices

            self._resolve_device_names()
            for serial in self.connected_devices:
                logger.info(f"Smartphone conectado: {self.device_names.get(serial, serial)} (Serial: {serial})")

            return len(self.connected_devices) > 0

        except Exception as e:
            logger.error(f"Erro ao conectar aos smartphones via ADB: {e}")
            return False

    def prepare_camera(self, device_id: str, preset_name: Optional[str] = None) -> bool:
        device_label = self.device_names.get(device_id, device_id)
        logger.info(f"Preparando aplicativo de câmera nativo em {device_label}...")

        if self.dry_run:
            logger.info(f"[DRY-RUN] Câmera preparada e pronta em {device_label}.")
            return True

        if not self.device_manager or device_id not in self.connected_devices:
            logger.error(f"Dispositivo {device_id} não está conectado.")
            return False

        try:
            actions = self.device_manager.get_device_actions(device_id)
            actions.turn_on_and_unlock_screen()
            actions.camera.open()
            time.sleep(1.0)
            return True
        except Exception as e:
            logger.error(f"Erro ao preparar câmera em {device_label}: {e}")
            return False

    def apply_preset(self, preset_name: str, device_id: Optional[str] = None) -> bool:
        # Modo default_app não suporta ajuste dinâmico de parâmetros de sensor
        logger.debug(f"[App Padrão ADB] Reconfiguração por preset ignorada (modo default_app).")
        return True

    def configure_camera(self, device_id: str, params: Dict[str, Any]) -> bool:
        logger.debug(f"[App Padrão ADB] Reconfiguração manual ignorada (modo default_app).")
        return True

    def capture_and_save_photo(
        self,
        device_id: str,
        target_dir: Path,
        target_filename: str
    ) -> Optional[Path]:
        device_label = self.device_names.get(device_id, device_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / target_filename

        logger.info(f"[App Padrão ADB] Capturando foto com {device_label} -> {final_path}")

        if self.dry_run:
            _create_simulated_image(final_path, device_label, target_filename, extra_text="Modo: App Padrão ADB (Simulado)")
            time.sleep(0.3)
            return final_path

        if not self.device_manager or device_id not in self.connected_devices:
            logger.error(f"Dispositivo {device_id} não está conectado.")
            return None

        try:
            actions = self.device_manager.get_device_actions(device_id)

            # Disparar foto
            actions.camera.take_picture()
            if self.config.capture_delay_sec > 0:
                time.sleep(self.config.capture_delay_sec)

            # Baixar a foto mais recente
            temp_download_dir = target_dir / "_temp"
            temp_download_dir.mkdir(parents=True, exist_ok=True)

            actions.camera.pull_pictures(destination=temp_download_dir, amount=1)

            downloaded_files = (
                list(temp_download_dir.glob("*.jpg")) +
                list(temp_download_dir.glob("*.jpeg")) +
                list(temp_download_dir.glob("*.png"))
            )
            if downloaded_files:
                latest_file = max(downloaded_files, key=lambda f: f.stat().st_mtime)
                shutil.move(str(latest_file), str(final_path))
                shutil.rmtree(str(temp_download_dir), ignore_errors=True)
                logger.info(f"Foto salva com sucesso em: {final_path}")
                return final_path
            else:
                logger.error(f"Nenhum arquivo baixado de {device_label} em {temp_download_dir}")
                shutil.rmtree(str(temp_download_dir), ignore_errors=True)
                return None

        except Exception as e:
            logger.error(f"Erro ao capturar/baixar foto de {device_label}: {e}")
            return None

    def get_last_telemetry(self, device_id: str) -> Optional[Dict[str, Any]]:
        return None

    def close_cameras(self) -> None:
        logger.info("Encerrando aplicativo de câmera nativo nos smartphones...")
        if self.dry_run or not self.device_manager:
            return

        for serial in self.connected_devices:
            try:
                actions = self.device_manager.get_device_actions(serial)
                actions.camera.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar câmera no serial {serial}: {e}")

    def _resolve_device_names(self) -> None:
        self.device_names.clear()
        for idx, serial in enumerate(self.connected_devices, start=1):
            if serial in self.config.camera_names:
                name = self.config.camera_names[serial]
            else:
                name = f"cam_{idx:02d}_{serial}"
            self.device_names[serial] = name


class SmartphoneController:
    """Controlador unificado para câmeras de smartphone (suporta 'ip_camera' e 'default_app')."""

    def __init__(self, config: SmartphoneConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

        mode_str = (config.mode or "ip_camera").lower().strip()
        if mode_str in ["adb", "default", "default_app", "stock", "oem"]:
            self.mode = "default_app"
            self.backend = ADBCameraBackend(config, dry_run=dry_run)
        else:
            self.mode = "ip_camera"
            self.backend = IPCameraBackend(config, dry_run=dry_run)

        logger.info(f"Controlador de Câmera inicializado no modo: [bold yellow]{self.mode}[/bold yellow]")

    @property
    def connected_devices(self) -> List[str]:
        return self.backend.connected_devices

    @property
    def connected_serials(self) -> List[str]:
        """Alias para manter compatibilidade com código existente."""
        return self.backend.connected_devices

    @property
    def connected_ips(self) -> List[str]:
        """Alias para manter compatibilidade com chamadas baseadas em IP."""
        return self.backend.connected_devices

    @property
    def device_names(self) -> Dict[str, str]:
        return self.backend.device_names

    def connect(self) -> bool:
        """Conecta aos dispositivos de câmera no modo configurado."""
        return self.backend.connect()

    def prepare_camera(self, device_id: str, preset_name: Optional[str] = None) -> bool:
        """Prepara o smartphone/câmera para captura."""
        return self.backend.prepare_camera(device_id, preset_name=preset_name)

    def apply_preset(self, preset_name: str, device_id: Optional[str] = None) -> bool:
        """Aplica um preset fotográfico nos dispositivos (efetivo no modo ip_camera)."""
        return self.backend.apply_preset(preset_name, device_id=device_id)

    def configure_camera(self, device_id: str, params: Dict[str, Any]) -> bool:
        """Aplica parâmetros fotográficos manuais dinamicamente."""
        return self.backend.configure_camera(device_id, params)

    def capture_and_save_photo(
        self,
        device_id: str,
        target_dir: Path,
        target_filename: str
    ) -> Optional[Path]:
        """Captura foto no dispositivo e salva no diretório especificado."""
        return self.backend.capture_and_save_photo(device_id, target_dir, target_filename)

    def get_last_telemetry(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Retorna os metadados de telemetria da última foto (foco, ISO, exposição)."""
        return self.backend.get_last_telemetry(device_id)

    def close_cameras(self) -> None:
        """Encerra a sessão com as câmeras."""
        self.backend.close_cameras()