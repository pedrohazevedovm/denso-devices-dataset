import argparse
import sys
from pathlib import Path

from src.visual_servoing_dataset.calibration.calibration_generator import (
    CharucoBoardSpec,
    generate_charuco_board_files,
)
from src.visual_servoing_dataset.calibration.calibration_manager import CalibrationManager
from src.visual_servoing_dataset.collector.dataset_collector import DatasetCollector
from src.visual_servoing_dataset.collector.session_manager import SessionManager
from src.visual_servoing_dataset.config import AppConfig, load_config
from src.visual_servoing_dataset.robot.denso_controller import DensoRobotController
from src.visual_servoing_dataset.smartphone.phone_controller import SmartphoneController
from src.visual_servoing_dataset.utils.logger import console, logger


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Coletor de Dataset para Visual Servoing (Fase 0: Calibração 2 em 1 ChArUco + "
            "Protocolo Experimental em 3 Fases SE(3) + LHS) com Robô DENSO VS-050 e Cam-App Android."
        )
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Caminho do arquivo de configuração (.yaml ou .json)",
    )
    parser.add_argument(
        "--generate-charuco",
        action="store_true",
        help="Fase 0A: Gerar prancha ChArUco em alta resolução (assets/charuco_board.png) e charuco_spec.json.",
    )
    parser.add_argument(
        "--center-charuco",
        action="store_true",
        help=(
            "Fase 0-Pre: Centralizar automaticamente o ChArUco (via marcador central ID=8) e alinhar "
            "o paralelismo pelos perímetros das extremidades (15,16 vs 0,1)."
        ),
    )
    parser.add_argument(
        "--no-auto-center",
        action="store_true",
        help="Desabilitar a centralização automática em P_cal,0 no início de --calibrate-camera.",
    )
    parser.add_argument(
        "--calibrate-camera",
        action="store_true",
        help="Fase 0B: Executar calibração automatizada '2 em 1' (Intrínsecos K, D + Hand-Eye AX=XB em 24 poses).",
    )
    parser.add_argument(
        "--verify-poses",
        action="store_true",
        help="Inspecionar supervisionadamente as 24 poses da abóbada esférica durante --calibrate-camera.",
    )
    parser.add_argument(
        "--rectify-dataset",
        action="store_true",
        help="Retificar imagens de um dataset existente utilizando calibration_profile.json (ImageRectifier LUT).",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Caminho para o arquivo calibration_profile.json (usado em --explore-limits, --collect e --rectify-dataset).",
    )
    parser.add_argument(
        "--explore-limits",
        action="store_true",
        help="Fase 1: Definir P0, calibrar 'manual_calibrado' a partir de 'auto' e explorar limites [l, u] salvando session_plan.json.",
    )
    parser.add_argument(
        "--generate-plan",
        action="store_true",
        help="Fase 2A: Gerar todas as poses candidatas de P0..P7 em SE(3) + LHS salvando poses_plan.json (status: pending_approval).",
    )
    parser.add_argument(
        "--verify-plan",
        action="store_true",
        help="Fase 2B: Verificar poses geradas (com substituição rastreável de rejeições em P7) e congelar poses_plan.json com SHA-256.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Fase 3: Executar coleta automática pareada ('auto' + 'manual_calibrado') a partir do plano congelado.",
    )
    parser.add_argument(
        "--session-plan",
        type=str,
        default=None,
        help="Caminho customizado para o arquivo session_plan.json (Fase 1 -> Fase 2).",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Caminho customizado para o arquivo poses_plan.json (Fase 2 -> Fase 3).",
    )
    parser.add_argument(
        "--reject-poses",
        nargs="*",
        default=None,
        help="Lista opcional de pose_ids (ex: p7_lhs_003) a rejeitar e substituir durante --verify-plan.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Habilitar confirmação passo a passo no terminal durante --explore-limits e --verify-plan.",
    )
    parser.add_argument(
        "--test-robot",
        action="store_true",
        help="Testar apenas a conexão e leitura de pose do robô DENSO VS-050.",
    )
    parser.add_argument(
        "--test-phone",
        action="store_true",
        help="Testar conexão HTTP com o Cam-App, calibração automática e captura nas configurações 'auto' e 'manual_calibrado'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forçar execução em modo de simulação (sem exigir robô ou smartphone físicos).",
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

    session_plan_path = Path(args.session_plan) if args.session_plan else None
    poses_plan_path = Path(args.plan) if args.plan else None
    calib_file_path = Path(args.calibration_file) if args.calibration_file else None

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
                        f"test_{preset_name}.jpg",
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

    ran_phase = False

    # Fase 0A: Geração da prancha ChArUco
    if args.generate_charuco:
        console.rule("[bold magenta]Fase 0A: Geração da Prancha ChArUco[/bold magenta]")
        spec = CharucoBoardSpec(
            squares_x=config.calibration.charuco_squares_x,
            squares_y=config.calibration.charuco_squares_y,
            square_length_m=config.calibration.charuco_square_length_m,
            marker_length_m=config.calibration.charuco_marker_length_m,
            dictionary=config.calibration.charuco_dictionary,
        )
        img_out, spec_out = generate_charuco_board_files(
            spec=spec,
            output_image_path=Path("assets/charuco_board.png"),
            output_spec_path=Path("assets/charuco_spec.json"),
        )
        logger.info(f"Tabuleiro ChArUco salvo em: {img_out.resolve()}")
        logger.info(f"Especificação de impressão 1:1 salva em: {spec_out.resolve()}")
        ran_phase = True

    if args.no_auto_center:
        config.calibration.auto_center_p0 = False

    # Fase 0-Pre: Centralização automática standalone do ChArUco (se chamado sem --calibrate-camera)
    if args.center_charuco and not args.calibrate_camera:
        cal_mgr = CalibrationManager(config)
        centered_pose, _ = cal_mgr.auto_center_charuco()
        config.calibration.p_cal_0_pose.cartesian = centered_pose.to_list()
        ran_phase = True

    # Fase 0B: Calibração "2 em 1" (Intrínsecos + Hand-Eye via ChArUco em 24 poses)
    if args.calibrate_camera:
        if args.center_charuco:
            config.calibration.auto_center_p0 = True
        cal_mgr = CalibrationManager(config)
        _, saved_cal_path = cal_mgr.run_calibration(
            verify_poses=args.verify_poses,
            output_profile_path=calib_file_path,
        )
        if calib_file_path is None:
            calib_file_path = saved_cal_path
        ran_phase = True

    # Retificação standalone de dataset existente
    if args.rectify_dataset:
        target_cal = calib_file_path or (
            Path(config.calibration.calibration_file)
            if config.calibration.calibration_file
            else (Path(config.dataset.output_dir) / "calibration_profile.json")
        )
        CalibrationManager.rectify_dataset_directory(
            dataset_dir=Path(config.dataset.output_dir),
            calibration_profile_path=target_cal,
        )
        ran_phase = True

    session_mgr = SessionManager(config, calibration_file=calib_file_path)

    if args.explore_limits:
        session_mgr.explore_limits(output_path=session_plan_path, interactive=args.interactive)
        ran_phase = True

    if args.generate_plan:
        session_mgr.generate_plan(session_plan_path=session_plan_path, output_plan_path=poses_plan_path)
        ran_phase = True

    if args.verify_plan:
        session_mgr.verify_plan(
            plan_path=poses_plan_path,
            interactive=args.interactive,
            reject_pose_ids=args.reject_poses,
        )
        ran_phase = True

    if args.collect or not ran_phase:
        collector = DatasetCollector(
            config,
            plan_path=poses_plan_path,
            calibration_file=calib_file_path,
        )
        success = collector.run()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
