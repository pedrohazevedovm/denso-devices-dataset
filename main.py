import sys
import argparse
from pathlib import Path

from src.visual_servoing_dataset.config import load_config, AppConfig
from src.visual_servoing_dataset.robot.denso_controller import DensoRobotController
from src.visual_servoing_dataset.smartphone.phone_controller import SmartphoneController
from src.visual_servoing_dataset.collector.dataset_collector import DatasetCollector
from src.visual_servoing_dataset.utils.logger import logger, console


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coletor de Dataset para Visual Servoing com Robô DENSO VS-050 e Cam-App Android."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Caminho do arquivo de configuração (.yaml ou .json)"
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Executar a sessão completa de captura do dataset."
    )
    parser.add_argument(
        "--test-robot",
        action="store_true",
        help="Testar apenas a conexão e leitura de pose do robô DENSO VS-050."
    )
    parser.add_argument(
        "--test-phone",
        action="store_true",
        help="Testar conexão HTTP com o Cam-App, calibração automática e captura nas configurações 'auto' e 'manual_calibrado'."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forçar execução em modo de simulação (sem exigir robô ou smartphone físicos)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.warning(f"Arquivo de configuração '{config_path}' não encontrado. Usando configurações padrão.")
        config = AppConfig()
    else:
        try:
            config = load_config(config_path)
            logger.info(f"Configurações carregadas de: {config_path.resolve()}")
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo de configuração: {e}")
            sys.exit(1)

    if args.dry_run:
        config.dry_run = True

    if args.test_robot:
        console.rule("[bold blue]Teste do Robô DENSO VS-050[/bold blue]")
        robot = DensoRobotController(config.robot, dry_run=config.dry_run)
        if robot.connect():
            pose = robot.get_cartesian_pose()
            logger.info(f"Pose atual lida do robô: {pose}")
            robot.disconnect()
            logger.info("Teste do robô concluído com SUCESSO!")
        else:
            logger.error("Teste do robô FALHOU!")
            sys.exit(1)
        return

    if args.test_phone:
        console.rule("[bold blue]Teste do Cam-App Android (Auto + Manual Calibrado)[/bold blue]")
        phone = SmartphoneController(config.smartphone, dry_run=config.dry_run)
        if phone.connect():
            test_dir = Path(config.dataset.output_dir) / "_test_captures"
            phone.calibrate_manual_from_auto()
            for dev in phone.connected_devices:
                phone.prepare_camera(dev)
                safe_dev_name = phone.device_names.get(dev, dev.replace(".", "_").replace(":", "_"))
                for preset_name in config.smartphone.active_presets:
                    phone.apply_preset(preset_name, device_id=dev)
                    saved = phone.capture_and_save_photo(
                        dev,
                        test_dir / safe_dev_name / preset_name,
                        f"test_{preset_name}.jpg"
                    )
                    if saved:
                        telemetry = phone.get_last_telemetry(dev)
                        if telemetry:
                            logger.info(
                                f"Metadados ({preset_name}) de {safe_dev_name}: "
                                f"Foco={telemetry.get('lens_focus_distance')} D, "
                                f"ISO={telemetry.get('sensor_sensitivity')}, "
                                f"Exp={telemetry.get('exposure_time_ms')} ms, "
                                f"K={telemetry.get('camera_matrix_k')}"
                            )
            phone.close_cameras()
            logger.info("Teste do Cam-App concluído com SUCESSO!")
        else:
            logger.error("Teste do Cam-App FALHOU!")
            sys.exit(1)
        return

    collector = DatasetCollector(config)
    success = collector.run()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
