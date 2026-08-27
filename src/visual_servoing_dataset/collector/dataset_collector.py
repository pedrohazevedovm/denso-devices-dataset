import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config import AppConfig, Pose, Joint
from ..robot.denso_controller import DensoRobotController
from ..robot.trajectories import PoseVariationGenerator, PoseWaypoint
from ..smartphone.phone_controller import SmartphoneController
from ..utils.logger import logger, console


class DatasetCollector:
    """Orquestrador da coleta de dataset sincronizando Robô DENSO VS-050 e Smartphones Android com objetos Pose/Joint."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.robot = DensoRobotController(config.robot, dry_run=config.dry_run)
        self.smartphone = SmartphoneController(config.smartphone, dry_run=config.dry_run)
        self.trajectory_generator = PoseVariationGenerator(config.variations)
        self.output_dir = Path(config.dataset.output_dir)

    def run(self) -> bool:
        """Executa a rotina completa de coleta do dataset por Cenas e Variações de Poses."""
        console.rule("[bold green]Iniciando Coleta do Dataset Visual Servoing (Pose/Joint aether_rdk)[/bold green]")
        logger.info(f"Modo Dry-Run (Simulação): {'SIM' if self.config.dry_run else 'NÃO'}")
        logger.info(f"Diretório raiz de saída: {self.output_dir.resolve()}")

        # Criar diretório base
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Conectar Robô
        logger.info("--- [Passo 1/4] Conectando ao Robô DENSO VS-050 ---")
        if not self.robot.connect():
            logger.error("Falha ao conectar com o robô DENSO. Abortando coleta.")
            return False

        # 2. Conectar Smartphones
        logger.info("--- [Passo 2/4] Conectando aos Smartphones Android ---")
        if not self.smartphone.connect():
            logger.error("Falha ao conectar com os smartphones Android. Abortando coleta.")
            self.robot.disconnect()
            return False

        # 3. Preparar câmeras nos smartphones
        for serial in self.smartphone.connected_serials:
            self.smartphone.prepare_camera(serial)

        # 4. Iterar sobre as Cenas
        logger.info("--- [Passo 3/4] Executando Poses e Capturas de Cenas ---")
        metadata_scenes: Dict[str, Any] = {}

        try:
            for scene_idx, scene_cfg in enumerate(self.config.scenes, start=1):
                logger.info(f"\n==================== CENA {scene_idx:02d}: {scene_cfg.name} ====================")
                waypoints = self.trajectory_generator.generate_waypoints_for_base_pose(scene_cfg.p0_pose)
                
                p0_pose_obj: Pose = scene_cfg.p0_pose.to_pose()

                scene_metadata: Dict[str, Any] = {
                    "name": scene_cfg.name,
                    "description": scene_cfg.description,
                    "p0_base_pose": p0_pose_obj.to_list(),
                    "waypoints": []
                }

                for wp_idx, wp in enumerate(waypoints, start=1):
                    logger.info(f"\n-> Waypoint {wp_idx}/{len(waypoints)} [{wp.id}] - Tipo: {wp.variation_type}")
                    
                    # Movimentar Robô com o objeto Pose
                    move_success = self.robot.move_cartesian(wp.pose)
                    if not move_success:
                        logger.error(f"Falha ao mover robô para waypoint {wp.id}. Pulando...")
                        continue

                    # Ler objetos Pose e Joint reais medidos pelo robô
                    actual_pose: Pose = self.robot.get_cartesian_pose()
                    actual_joint: Joint = self.robot.get_joints_pose()

                    captures_info: Dict[str, str] = {}

                    # Capturar foto em cada smartphone conectado
                    for serial in self.smartphone.connected_serials:
                        cam_folder_name = self.smartphone.device_names.get(serial, serial)
                        target_cam_dir = self.output_dir / scene_cfg.name / cam_folder_name
                        
                        saved_path = self.smartphone.capture_and_save_photo(
                            serial=serial,
                            target_dir=target_cam_dir,
                            target_filename=wp.filename
                        )

                        if saved_path:
                            rel_path = saved_path.relative_to(self.output_dir).as_posix()
                            captures_info[cam_folder_name] = rel_path

                    # Registrar metadados do waypoint usando as listas dos objetos Pose e Joint
                    wp_data = {
                        "waypoint_id": wp.id,
                        "filename": wp.filename,
                        "variation_type": wp.variation_type,
                        "commanded_cartesian_pose": wp.pose.to_list(),
                        "actual_cartesian_pose": actual_pose.to_list(),
                        "actual_joint_pose": actual_joint.to_list(),
                        "delta_applied": wp.delta_applied,
                        "timestamp": datetime.now().isoformat(),
                        "captures": captures_info
                    }
                    scene_metadata["waypoints"].append(wp_data)

                metadata_scenes[scene_cfg.name] = scene_metadata

            # 5. Salvar metadata.json unificado na raiz do dataset
            logger.info("--- [Passo 4/4] Gerando metadata.json do Dataset ---")
            full_metadata = {
                "dataset_version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "dry_run": self.config.dry_run,
                "robot_info": {
                    "model": "DENSO VS-050",
                    "ip_address": self.config.robot.ip_address,
                    "tcp_offset": self.config.robot.tcp_offset.to_dict()
                },
                "variations_config": {
                    "delta_trans_xy": self.config.variations.delta_trans_xy,
                    "delta_rot_z": self.config.variations.delta_rot_z,
                    "delta_scale_z": self.config.variations.delta_scale_z,
                    "delta_combined": self.config.variations.delta_combined
                },
                "cameras": self.smartphone.device_names,
                "scenes": metadata_scenes
            }

            metadata_file = self.output_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(full_metadata, f, indent=2, ensure_ascii=False)

            console.rule("[bold green]Coleta Concluída com Sucesso![/bold green]")
            logger.info(f"Metadados salvos em: {metadata_file.resolve()}")
            return True

        except KeyboardInterrupt:
            logger.warning("Coleta interrompida manualmente pelo usuário (Ctrl+C).")
            return False

        except Exception as e:
            logger.error(f"Exceção durante a coleta do dataset: {e}")
            return False

        finally:
            self.smartphone.close_cameras()
            self.robot.disconnect()
