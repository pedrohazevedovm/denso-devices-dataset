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
        description="Coletor de Dataset para Visual Servoing com Robô DENSO VS-050 e Smartphones Android."
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
        help="Testar apenas a conexão e movimentação básica do robô DENSO VS-050."
    )
    parser.add_argument(
        "--test-phone",
        action="store_true",
        help="Testar apenas a conexão ADB e foto dos smartphones Android."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forçar execução em modo de simulação (sem exigir robô ou smartphone físicos)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Carregar arquivo de configuração
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

    # Tratar modos de teste standalone
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
        console.rule("[bold blue]Teste dos Smartphones Android[/bold blue]")
        phone = SmartphoneController(config.smartphone, dry_run=config.dry_run)
        if phone.connect():
            test_dir = Path(config.dataset.output_dir) / "_test_captures"
            for serial in phone.connected_ips:
                phone.prepare_camera(serial)
                phone.capture_and_save_photo(serial, test_dir, f"test_{serial}.jpg")
            phone.close_cameras()
            logger.info("Teste de smartphone concluído com SUCESSO!")
        else:
            logger.error("Teste de smartphone FALHOU!")
            sys.exit(1)
        return

    # Execução Padrão: Coleta do Dataset
    collector = DatasetCollector(config)
    success = collector.run()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
