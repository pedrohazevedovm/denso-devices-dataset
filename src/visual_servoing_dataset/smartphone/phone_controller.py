import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont

from ..config import SmartphoneConfig
from ..utils.logger import logger

try:
    from device_manager import DeviceManager
    DEVICE_MANAGER_AVAILABLE = True
except ImportError:
    DEVICE_MANAGER_AVAILABLE = False
    DeviceManager = None


class SmartphoneController:
    """Controlador para smartphones Android via device_manager (ADB)."""

    def __init__(self, config: SmartphoneConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run or not DEVICE_MANAGER_AVAILABLE
        self.device_manager: Optional[Any] = None
        self.connected_serials: List[str] = []
        self.device_names: Dict[str, str] = {}  # serial -> readable name (ex: "cam_01_moto_edge_40")

    def connect(self) -> bool:
        """Descobre e conecta aos dispositivos Android via USB/ADB ou IP."""
        if self.dry_run:
            logger.info("[DRY-RUN] Simulação de conexão com smartphones Android.")
            if self.config.serials and "auto" not in self.config.serials:
                self.connected_serials = self.config.serials
            else:
                self.connected_serials = ["sim_serial_01", "sim_serial_02"]

            self._resolve_device_names()
            for serial, name in self.device_names.items():
                logger.info(f"[DRY-RUN] Smartphone simulado: {name} (Serial: {serial})")
            return True

        if not DEVICE_MANAGER_AVAILABLE:
            logger.error("A biblioteca 'device_manager' não está disponível!")
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
                self.connected_serials = self.device_manager.connect_devices(*target_serials)
            else:
                self.connected_serials = self.device_manager.connected_devices

            self._resolve_device_names()
            for serial in self.connected_serials:
                logger.info(f"Smartphone conectado: {self.device_names.get(serial, serial)} (Serial: {serial})")

            return len(self.connected_serials) > 0

        except Exception as e:
            logger.error(f"Erro ao conectar aos smartphones via ADB: {e}")
            return False

    def prepare_camera(self, serial: str) -> bool:
        """Desbloqueia a tela e abre o aplicativo de câmera no smartphone especificado."""
        device_label = self.device_names.get(serial, serial)
        logger.info(f"Preparando câmera no smartphone: {device_label}")

        if self.dry_run:
            logger.info(f"[DRY-RUN] Câmera preparada e pronta em {device_label}.")
            return True

        if not self.device_manager or serial not in self.connected_serials:
            logger.error(f"Dispositivo {serial} não está conectado.")
            return False

        try:
            actions = self.device_manager.get_device_actions(serial)
            actions.turn_on_and_unlock_screen()
            actions.camera.open()
            time.sleep(1.0)
            return True
        except Exception as e:
            logger.error(f"Erro ao preparar câmera em {device_label}: {e}")
            return False

    def capture_and_save_photo(
        self,
        serial: str,
        target_dir: Path,
        target_filename: str
    ) -> Optional[Path]:
        """Dispara a foto no smartphone e salva no diretório de destino local."""
        device_label = self.device_names.get(serial, serial)
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / target_filename

        logger.info(f"Capturando foto com {device_label} -> {final_path}")

        if self.dry_run:
            self._create_simulated_image(final_path, device_label, target_filename)
            time.sleep(0.3)
            return final_path

        if not self.device_manager or serial not in self.connected_serials:
            logger.error(f"Dispositivo {serial} não está conectado.")
            return None

        try:
            actions = self.device_manager.get_device_actions(serial)
            
            # Disparar foto
            actions.camera.take_picture()
            if self.config.capture_delay_sec > 0:
                time.sleep(self.config.capture_delay_sec)

            # Baixar a foto mais recente
            temp_download_dir = target_dir / "_temp"
            temp_download_dir.mkdir(parents=True, exist_ok=True)
            
            actions.camera.pull_pictures(destination=temp_download_dir, amount=1)
            
            # Encontrar arquivo baixado no diretório temporário
            downloaded_files = list(temp_download_dir.glob("*.jpg")) + list(temp_download_dir.glob("*.jpeg")) + list(temp_download_dir.glob("*.png"))
            if downloaded_files:
                # Pegar o mais recente
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

    def close_cameras(self) -> None:
        """Fecha o app de câmera em todos os dispositivos conectados."""
        logger.info("Encerrando aplicativo de câmera nos smartphones...")
        if self.dry_run or not self.device_manager:
            return

        for serial in self.connected_serials:
            try:
                actions = self.device_manager.get_device_actions(serial)
                actions.camera.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar câmera no serial {serial}: {e}")

    def _resolve_device_names(self) -> None:
        """Resolve nomes amigáveis para as pastas dos dispositivos (ex: cam_01_moto_edge_40)."""
        self.device_names.clear()
        for idx, serial in enumerate(self.connected_serials, start=1):
            if serial in self.config.camera_names:
                name = self.config.camera_names[serial]
            else:
                name = f"cam_{idx:02d}_{serial}"
            self.device_names[serial] = name

    def _create_simulated_image(self, target_path: Path, camera_name: str, filename: str) -> None:
        """Gera uma imagem sintética para uso no modo dry-run / simulação."""
        img = Image.new("RGB", (640, 480), color=(40, 44, 52))
        draw = ImageDraw.Draw(img)
        
        # Desenhar borda e texto demonstrativo
        draw.rectangle([(10, 10), (630, 470)], outline=(0, 200, 255), width=3)
        draw.text((30, 30), f"Camera: {camera_name}", fill=(255, 255, 255))
        draw.text((30, 60), f"File: {filename}", fill=(0, 255, 128))
        draw.text((30, 90), f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", fill=(200, 200, 200))
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(target_path, "JPEG")