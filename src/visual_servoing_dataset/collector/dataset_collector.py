"""Orquestrador da Fase 3: Coleta Automática Pareada em SE(3) + Calibração Hand-Eye + Retificação LUT + QA."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ..calibration.calibration_solver import (
    ImageRectifier,
    compute_planar_homography_gt,
    compute_plane_normal_and_distance_in_cam,
)
from ..config import (
    AUTO_MODE_DESCRIPTION,
    MANUAL_CALIBRADO_DESCRIPTION,
    AppConfig,
    Joint,
    Pose,
    ensure_pose,
)
from ..robot.denso_controller import DensoRobotController
from ..robot.kinematics import (
    compute_pose_error,
    compute_relative_transform,
    invert_se3,
    matrix_to_row_major_list,
    pose_to_matrix,
    tcp_offset_to_matrix,
)
from ..schemas import (
    CalibrationProfileDocument,
    PlannedPoseCandidate,
    PosesPlanDocument,
    audit_camera_parameters,
    compute_file_sha256,
)
from ..smartphone.phone_controller import SmartphoneController
from ..utils.logger import console, logger
from .session_manager import SessionManager


def _inspect_image_integrity(filepath: Path, expected_camera_id: str) -> Dict[str, Any]:
    """Valida a integridade física do arquivo JPEG, resolução, orientação e camera_physical_id."""
    if not filepath.exists():
        return {"valid": False, "error": "file_missing"}
    size_bytes = filepath.stat().st_size
    if size_bytes <= 0:
        return {"valid": False, "error": "empty_file", "size_bytes": 0}

    try:
        with Image.open(filepath) as img:
            width, height = img.size
            fmt = img.format
        is_landscape = width >= height
        return {
            "valid": bool(width > 0 and height > 0 and fmt == "JPEG"),
            "size_bytes": size_bytes,
            "format": fmt,
            "width": width,
            "height": height,
            "orientation": "landscape_0deg" if is_landscape else "portrait",
            "camera_physical_id": expected_camera_id,
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc), "size_bytes": size_bytes}


class DatasetCollector:
    """Orquestrador da Fase 3: executa a coleta pareada ('auto' + 'manual_calibrado') sobre poses_plan.json."""

    def __init__(
        self,
        config: AppConfig,
        plan_path: Optional[Path] = None,
        calibration_file: Optional[Path] = None,
    ):
        self.config = config
        self.euler_convention = config.robot.euler_convention
        self.robot = DensoRobotController(config.robot, dry_run=config.dry_run)
        self.smartphone = SmartphoneController(config.smartphone, dry_run=config.dry_run)
        self.output_dir = Path(config.dataset.output_dir)
        self.plan_path = plan_path or (self.output_dir / config.dataset.poses_plan_file)

        resolved_cal_file = calibration_file or (
            Path(config.calibration.calibration_file)
            if config.calibration.calibration_file
            else (self.output_dir / "calibration_profile.json")
        )
        self.calibration_file: Optional[Path] = (
            resolved_cal_file if (resolved_cal_file and resolved_cal_file.exists()) else None
        )
        self.calibration_profile: Optional[CalibrationProfileDocument] = None
        self.rectifier: Optional[ImageRectifier] = None
        if self.calibration_file is not None:
            self._load_calibration_profile(self.calibration_file)

        self.session_manager = SessionManager(config, calibration_file=self.calibration_file)

    def _load_calibration_profile(self, path: Path) -> None:
        """Carrega calibration_profile.json e inicializa o ImageRectifier por LUT fixa."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_cal = json.load(f)
            self.calibration_profile = CalibrationProfileDocument.from_dict(raw_cal)
            self.rectifier = ImageRectifier(
                k_matrix=self.calibration_profile.intrinsics["K"],
                distortion_coeffs=self.calibration_profile.intrinsics["distortion_coeffs"],
                resolution_px=(
                    self.calibration_profile.resolution_px[0],
                    self.calibration_profile.resolution_px[1],
                ),
                k_rectified=self.calibration_profile.intrinsics.get("K_rectified"),
            )
            logger.info(
                f"Perfil de calibração Fase 0 carregado: {self.calibration_profile.calibration_id} "
                f"({path.resolve()})"
            )
        except Exception as exc:
            logger.warning(f"Não foi possível carregar calibration_profile.json de '{path}': {exc}")
            self.calibration_profile = None
            self.rectifier = None

    def _load_or_prepare_frozen_plan(self) -> PosesPlanDocument:
        """Carrega o plano congelado poses_plan.json (ou gera/congela em modo automático caso não exista)."""
        if not self.plan_path.exists():
            logger.info(
                f"Plano '{self.plan_path}' não encontrado. Executando geração e verificação automática do plano..."
            )
            return self.session_manager.verify_plan(plan_path=self.plan_path, interactive=False)

        with open(self.plan_path, "r", encoding="utf-8") as f:
            raw_doc = json.load(f)

        plan_doc = PosesPlanDocument.from_dict(raw_doc)
        if plan_doc.status != "frozen":
            logger.warning(
                f"Plano '{self.plan_path}' está com status='{plan_doc.status}'. Congelando plano antes da coleta..."
            )
            return self.session_manager.verify_plan(plan_path=self.plan_path, interactive=False)

        return plan_doc

    def run(self) -> bool:
        """Executa a Fase 3 completa sobre todas as poses aprovadas no plano congelado."""
        console.rule(
            "[bold green]Fase 3: Coleta Automática Pareada em SE(3) (DENSO VS-050 + Cam-App)[/bold green]"
        )
        logger.info(f"Modo Dry-Run (Simulação): {'SIM' if self.config.dry_run else 'NÃO'}")
        logger.info(f"Diretório raiz de saída: {self.output_dir.resolve()}")
        logger.info(f"Definição manual_calibrado: {MANUAL_CALIBRADO_DESCRIPTION}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Verificar novamente se calibration_profile.json foi criado no diretório de saída
        if self.calibration_profile is None:
            candidate_cal = self.output_dir / "calibration_profile.json"
            if candidate_cal.exists():
                self.calibration_file = candidate_cal
                self._load_calibration_profile(candidate_cal)

        plan_doc = self._load_or_prepare_frozen_plan()

        # 1. Conectar Robô
        logger.info("--- [Passo 1/4] Conectando ao Robô DENSO VS-050 ---")
        if not self.robot.connect():
            logger.error("Falha ao conectar com o robô DENSO. Abortando coleta.")
            return False

        # 2. Conectar Smartphones (Cam-App via HTTP)
        logger.info("--- [Passo 2/4] Conectando ao Cam-App Android via HTTP ---")
        if not self.smartphone.connect():
            logger.error("Falha ao conectar com o Cam-App nos smartphones. Abortando coleta.")
            self.robot.disconnect()
            return False

        for device_id in self.smartphone.connected_devices:
            self.smartphone.prepare_camera(device_id)

        # 3. Executar Coleta Pareada por Cena
        logger.info("--- [Passo 3/4] Executando Coleta Pareada nas Poses Aprovadas (P0..P7) ---")
        metadata_scenes: Dict[str, Any] = {}
        tabular_rows: List[Dict[str, Any]] = []

        tcp_mat = tcp_offset_to_matrix(self.config.robot.tcp_offset, euler_convention=self.euler_convention)
        tcp_mat_list = matrix_to_row_major_list(tcp_mat)

        # Extrair matriz Hand-Eye ^T T_C (em mm) e pose do plano na base se calibration_profile estiver ativo
        t_tcp_to_cam_mm = np.eye(4, dtype=np.float64)
        t_base_to_board_mm: Optional[np.ndarray] = None
        locked_calib_focus: Optional[float] = None
        if self.calibration_profile is not None:
            he = self.calibration_profile.hand_eye
            if "T_tcp_approx_to_camera_mm" in he:
                t_tcp_to_cam_mm = np.asarray(he["T_tcp_approx_to_camera_mm"], dtype=np.float64)
            elif "T_tcp_approx_to_camera" in he:
                t_tcp_to_cam_mm = np.asarray(he["T_tcp_approx_to_camera"], dtype=np.float64).copy()
                t_tcp_to_cam_mm[:3, 3] *= 1000.0

            if "T_base_to_board_mean_mm" in he:
                t_base_to_board_mm = np.asarray(he["T_base_to_board_mean_mm"], dtype=np.float64)
            elif "T_base_to_board_mean" in he:
                t_base_to_board_mm = np.asarray(he["T_base_to_board_mean"], dtype=np.float64).copy()
                t_base_to_board_mm[:3, 3] *= 1000.0

            locked_calib_focus = self.calibration_profile.locked_camera_params.get("focus_distance_diopters")

        try:
            for scene_idx, (scene_id, scene_plan) in enumerate(plan_doc.scenes.items(), start=1):
                logger.info(
                    f"\n==================== CENA {scene_idx:02d}: {scene_id} ({scene_plan.scene_category}) ===================="
                )
                p0_pose: Pose = ensure_pose(scene_plan.p0_cartesian)
                t_b_t0 = pose_to_matrix(p0_pose, euler_convention=self.euler_convention)
                t_b_c0 = t_b_t0 @ t_tcp_to_cam_mm

                # Posicionar em P0 e calibrar 'manual_calibrado' a partir de 'auto' no início da cena
                self.robot.move_cartesian(p0_pose)
                calibrated_raw = self.smartphone.calibrate_manual_from_auto()

                # Se houver calibration_profile ativo, travar foco no valor calibrado na Fase 0
                if locked_calib_focus is not None:
                    for dev_id, cal_cfg in calibrated_raw.items():
                        auto_f = float(cal_cfg.get("focus_distance", locked_calib_focus))
                        if abs(auto_f - float(locked_calib_focus)) > 0.05:
                            logger.warning(
                                f"[AUDITORIA CALIBRAÇÃO] Foco em P0 ({auto_f:.4f} D) difere do foco calibrado "
                                f"({locked_calib_focus:.4f} D). Travando focus_distance={locked_calib_focus:.4f} D!"
                            )
                        cal_cfg["focus_distance"] = float(locked_calib_focus)
                        self.smartphone.get_device_presets(dev_id)["manual_calibrado"]["focus_distance"] = float(
                            locked_calib_focus
                        )

                calibrated_by_cam = {
                    self.smartphone.device_names.get(dev, dev): cfg
                    for dev, cfg in calibrated_raw.items()
                }

                approved_poses: List[PlannedPoseCandidate] = [
                    p for p in scene_plan.poses if p.status == "approved"
                ]
                rejected_poses: List[PlannedPoseCandidate] = [
                    p for p in scene_plan.poses if p.status == "rejected"
                ]
                logger.info(
                    f"[{scene_id}] Total de poses aprovadas para coleta: {len(approved_poses)} "
                    f"(rejeitadas preservadas no histórico: {len(rejected_poses)})"
                )

                executed_pairs: List[Dict[str, Any]] = []

                for wp_idx, pose_cand in enumerate(approved_poses, start=1):
                    logger.info(
                        f"\n-> [{scene_id}] Pose {wp_idx}/{len(approved_poses)} "
                        f"| ID: {pose_cand.pose_id} ({pose_cand.pose_family}) | Tipo: {pose_cand.variation_type}"
                    )
                    target_pose_obj = ensure_pose(pose_cand.target_cartesian_pose)

                    # 1 & 2. Posicionamento + Estabilização Mecânica
                    if not self.robot.move_cartesian(target_pose_obj):
                        logger.error(f"Falha ao mover robô para pose {pose_cand.pose_id}. Pulando...")
                        continue

                    # 3. Leitura Pré-Captura
                    timestamp_robot_before_ns = time.time_ns()
                    pose_before: Pose = self.robot.get_cartesian_pose()
                    joints_before: Joint = self.robot.get_joints_pose()
                    t_b_tk_before = pose_to_matrix(pose_before, euler_convention=self.euler_convention)
                    delta_before_mat, xi_before = compute_relative_transform(
                        t_b_t0, t_b_tk_before, euler_convention=self.euler_convention
                    )

                    # Pose óptica real da câmera na base: ^B T_C(k) = ^B T_T(k) * ^T T_C
                    t_b_ck_before = t_b_tk_before @ t_tcp_to_cam_mm
                    delta_cam_before_mat = invert_se3(t_b_c0) @ t_b_ck_before

                    # Homografia teórica H_gt em relação a P0 (quando calibração Fase 0 está disponível)
                    h_gt_list: Optional[List[List[float]]] = None
                    if self.rectifier is not None and t_base_to_board_mm is not None:
                        t_ck_c0 = invert_se3(t_b_ck_before) @ t_b_c0
                        n_c0, d_c0 = compute_plane_normal_and_distance_in_cam(t_b_c0, t_base_to_board_mm)
                        h_gt_mat = compute_planar_homography_gt(
                            self.rectifier.k_rectified, t_ck_c0, n_c0, d_c0
                        )
                        h_gt_list = matrix_to_row_major_list(h_gt_mat)

                    # 4 & 5. Captura Pareada ('auto' seguida de 'manual_calibrado' SEM mover o robô)
                    captures_by_camera: Dict[str, Dict[str, Any]] = {}

                    for device_id in self.smartphone.connected_devices:
                        cam_folder_name = self.smartphone.device_names.get(device_id, device_id)
                        captures_by_camera[cam_folder_name] = {}

                        for preset_name in ("auto", "manual_calibrado"):
                            if preset_name not in self.config.smartphone.active_presets:
                                continue

                            self.smartphone.apply_preset(preset_name, device_id=device_id)
                            target_cam_dir = self.output_dir / scene_id / cam_folder_name / preset_name
                            timestamp_host_ns = time.time_ns()

                            saved_path = self.smartphone.capture_and_save_photo(
                                device_id=device_id,
                                target_dir=target_cam_dir,
                                target_filename=pose_cand.filename,
                            )

                            if saved_path and saved_path.exists():
                                rel_path = saved_path.relative_to(self.output_dir).as_posix()
                                file_sha256 = compute_file_sha256(saved_path)
                                effective_meta = self.smartphone.get_last_telemetry(device_id) or {}
                                applied_settings = self.smartphone.get_last_applied_settings(device_id) or {}
                                requested_preset = self.smartphone.get_device_presets(device_id).get(preset_name, {})

                                param_audit = audit_camera_parameters(
                                    preset_name=preset_name,
                                    requested_settings=applied_settings or requested_preset,
                                    effective_metadata=effective_meta,
                                )
                                img_qa = _inspect_image_integrity(
                                    saved_path,
                                    expected_camera_id=self.config.smartphone.camera_physical_id,
                                )
                                timestamp_cam_ns = int(
                                    effective_meta.get("sensor_timestamp", timestamp_host_ns)
                                )

                                # Se ImageRectifier estiver ativo, salvar também imagem retificada em rectified/
                                rect_rel_path: Optional[str] = None
                                rect_sha256: Optional[str] = None
                                if self.rectifier is not None and self.config.calibration.rectify_on_collect:
                                    raw_img = cv2.imread(str(saved_path), cv2.IMREAD_COLOR)
                                    if raw_img is not None:
                                        rect_img = self.rectifier.rectify_image(raw_img)
                                        rect_path = (
                                            self.output_dir
                                            / scene_id
                                            / cam_folder_name
                                            / "rectified"
                                            / preset_name
                                            / pose_cand.filename
                                        )
                                        rect_path.parent.mkdir(parents=True, exist_ok=True)
                                        cv2.imwrite(str(rect_path), rect_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                                        rect_rel_path = rect_path.relative_to(self.output_dir).as_posix()
                                        rect_sha256 = compute_file_sha256(rect_path)

                                mode_record = {
                                    "preset": preset_name,
                                    "mode_description": (
                                        AUTO_MODE_DESCRIPTION
                                        if preset_name == "auto"
                                        else MANUAL_CALIBRADO_DESCRIPTION
                                    ),
                                    "relative_path": rel_path,
                                    "sha256": file_sha256,
                                    "rectified_relative_path": rect_rel_path,
                                    "rectified_sha256": rect_sha256,
                                    "timestamp_camera_ns": timestamp_cam_ns,
                                    "timestamp_host_ns": timestamp_host_ns,
                                    "requested_settings": requested_preset,
                                    "applied_settings": applied_settings,
                                    "effective_metadata": effective_meta,
                                    "parameter_audit": param_audit,
                                    "image_qa": img_qa,
                                }
                                captures_by_camera[cam_folder_name][preset_name] = mode_record

                    # 6. Leitura Pós-Captura e QA de Estabilidade Mecânica
                    timestamp_robot_after_ns = time.time_ns()
                    pose_after: Pose = self.robot.get_cartesian_pose()
                    joints_after: Joint = self.robot.get_joints_pose()
                    t_b_tk_after = pose_to_matrix(pose_after, euler_convention=self.euler_convention)
                    t_b_ck_after = t_b_tk_after @ t_tcp_to_cam_mm
                    delta_after_mat, xi_after = compute_relative_transform(
                        t_b_t0, t_b_tk_after, euler_convention=self.euler_convention
                    )
                    drift_mm, drift_deg = compute_pose_error(
                        pose_before, pose_after, euler_convention=self.euler_convention
                    )
                    pos_err_mm, pos_err_deg = compute_pose_error(
                        target_pose_obj, pose_before, euler_convention=self.euler_convention
                    )

                    pair_record = {
                        "pair_id": pose_cand.pair_id,
                        "pose_id": pose_cand.pose_id,
                        "pose_family": pose_cand.pose_family,
                        "variation_type": pose_cand.variation_type,
                        "repeat_index": pose_cand.repeat_index,
                        "filename": pose_cand.filename,
                        "replaces_pose_id": pose_cand.replaces_pose_id,
                        "replacement_seed": pose_cand.replacement_seed,
                        "robot_geometry": {
                            "euler_convention": self.euler_convention,
                            "tcp_flange_to_tcp_pose": self.config.robot.tcp_offset.to_dict(),
                            "tcp_flange_to_tcp_matrix_4x4": tcp_mat_list,
                            "p0_reference_pose": [round(float(v), 6) for v in p0_pose.to_list()],
                            "p0_reference_matrix_4x4": scene_plan.p0_matrix_4x4,
                            "target_cartesian_pose": pose_cand.target_cartesian_pose,
                            "target_matrix_4x4": pose_cand.target_matrix_4x4,
                            "pose_measured_before": [round(float(v), 6) for v in pose_before.to_list()],
                            "pose_measured_before_matrix_4x4": matrix_to_row_major_list(t_b_tk_before),
                            "camera_optical_pose_before_matrix_4x4": matrix_to_row_major_list(t_b_ck_before),
                            "joints_measured_before": [round(float(v), 6) for v in joints_before.to_list()],
                            "timestamp_robot_before_ns": timestamp_robot_before_ns,
                            "pose_measured_after": [round(float(v), 6) for v in pose_after.to_list()],
                            "pose_measured_after_matrix_4x4": matrix_to_row_major_list(t_b_tk_after),
                            "camera_optical_pose_after_matrix_4x4": matrix_to_row_major_list(t_b_ck_after),
                            "joints_measured_after": [round(float(v), 6) for v in joints_after.to_list()],
                            "timestamp_robot_after_ns": timestamp_robot_after_ns,
                            "xi_commanded_local": pose_cand.xi_local,
                            "delta_t_commanded_4x4": pose_cand.delta_t_local_4x4,
                            "xi_measured_before_local": [round(float(v), 6) for v in xi_before],
                            "delta_t_measured_before_4x4": matrix_to_row_major_list(delta_before_mat),
                            "delta_t_camera_optical_before_4x4": matrix_to_row_major_list(delta_cam_before_mat),
                            "xi_measured_after_local": [round(float(v), 6) for v in xi_after],
                            "delta_t_measured_after_4x4": matrix_to_row_major_list(delta_after_mat),
                            "homography_gt_from_p0_3x3": h_gt_list,
                            "positioning_error_mm": round(pos_err_mm, 6),
                            "positioning_error_deg": round(pos_err_deg, 6),
                            "inter_capture_drift_mm": round(drift_mm, 6),
                            "inter_capture_drift_deg": round(drift_deg, 6),
                        },
                        "captures": captures_by_camera,
                    }
                    executed_pairs.append(pair_record)

                    # Adicionar linhas para o registro tabular (.csv) por quadro
                    for cam_name, modes_dict in captures_by_camera.items():
                        for mode_name, mode_data in modes_dict.items():
                            eff = mode_data.get("effective_metadata", {})
                            audit = mode_data.get("parameter_audit", {})
                            qa = mode_data.get("image_qa", {})
                            tabular_rows.append({
                                "dataset_version": self.config.dataset.dataset_version,
                                "schema_version": self.config.dataset.schema_version,
                                "session_id": self.config.dataset.session_id,
                                "calibration_id": (
                                    self.calibration_profile.calibration_id
                                    if self.calibration_profile
                                    else None
                                ),
                                "scene_id": scene_id,
                                "scene_category": scene_plan.scene_category,
                                "smartphone_id": self.config.smartphone.smartphone_id,
                                "camera_name": cam_name,
                                "camera_physical_id": self.config.smartphone.camera_physical_id,
                                "mount_id": self.config.robot.mount_id,
                                "tool_definition_id": self.config.robot.tool_definition_id,
                                "pair_id": pose_cand.pair_id,
                                "pose_id": pose_cand.pose_id,
                                "pose_family": pose_cand.pose_family,
                                "variation_type": pose_cand.variation_type,
                                "repeat_index": pose_cand.repeat_index,
                                "mode": mode_name,
                                "relative_path": mode_data.get("relative_path"),
                                "sha256": mode_data.get("sha256"),
                                "rectified_relative_path": mode_data.get("rectified_relative_path"),
                                "rectified_sha256": mode_data.get("rectified_sha256"),
                                "timestamp_camera_ns": mode_data.get("timestamp_camera_ns"),
                                "timestamp_host_ns": mode_data.get("timestamp_host_ns"),
                                "timestamp_robot_before_ns": timestamp_robot_before_ns,
                                "timestamp_robot_after_ns": timestamp_robot_after_ns,
                                "xi_dx_mm": pose_cand.xi_local[0],
                                "xi_dy_mm": pose_cand.xi_local[1],
                                "xi_dz_mm": pose_cand.xi_local[2],
                                "xi_drx_deg": pose_cand.xi_local[3],
                                "xi_dry_deg": pose_cand.xi_local[4],
                                "xi_drz_deg": pose_cand.xi_local[5],
                                "measured_dx_mm": round(float(xi_before[0]), 6),
                                "measured_dy_mm": round(float(xi_before[1]), 6),
                                "measured_dz_mm": round(float(xi_before[2]), 6),
                                "measured_drx_deg": round(float(xi_before[3]), 6),
                                "measured_dry_deg": round(float(xi_before[4]), 6),
                                "measured_drz_deg": round(float(xi_before[5]), 6),
                                "inter_capture_drift_mm": round(drift_mm, 6),
                                "inter_capture_drift_deg": round(drift_deg, 6),
                                "lens_focus_distance": eff.get("lens_focus_distance"),
                                "sensor_sensitivity": eff.get("sensor_sensitivity"),
                                "sensor_exposure_time_ns": eff.get("sensor_exposure_time"),
                                "lens_state": eff.get("lens_state"),
                                "ois_mode": eff.get("lens_optical_stabilization_mode"),
                                "eis_mode": eff.get("control_video_stabilization_mode"),
                                "audit_compliant": audit.get("compliant"),
                                "image_valid": qa.get("valid"),
                                "image_width": qa.get("width"),
                                "image_height": qa.get("height"),
                            })

                # Verificar deriva entre P0 inicial e P0 final da cena
                p0_drift_summary: Optional[Dict[str, float]] = None
                if len(executed_pairs) >= 2:
                    first_pair = executed_pairs[0]
                    last_pair = executed_pairs[-1]
                    if (
                        first_pair["variation_type"] == "reference_start"
                        and last_pair["variation_type"] == "reference_end"
                    ):
                        p_start = ensure_pose(first_pair["robot_geometry"]["pose_measured_before"])
                        p_end = ensure_pose(last_pair["robot_geometry"]["pose_measured_before"])
                        d_mm, d_deg = compute_pose_error(p_start, p_end, euler_convention=self.euler_convention)
                        p0_drift_summary = {
                            "start_to_end_trans_drift_mm": round(d_mm, 6),
                            "start_to_end_rot_drift_deg": round(d_deg, 6),
                        }

                metadata_scenes[scene_id] = {
                    "scene_id": scene_id,
                    "scene_category": scene_plan.scene_category,
                    "description": scene_plan.description,
                    "euler_convention": self.euler_convention,
                    "p0_base_pose": scene_plan.p0_cartesian,
                    "p0_base_joints": scene_plan.p0_joints,
                    "p0_base_matrix_4x4": scene_plan.p0_matrix_4x4,
                    "lower_bounds": scene_plan.lower_bounds,
                    "upper_bounds": scene_plan.upper_bounds,
                    "isolated_fractions": scene_plan.isolated_fractions,
                    "calibrated_manual_settings": calibrated_by_cam,
                    "lhs_metadata": scene_plan.lhs_metadata.to_dict(),
                    "p0_start_end_drift": p0_drift_summary,
                    "rejected_poses_history": [p.to_dict() for p in rejected_poses],
                    "pairs": executed_pairs,
                }

            # 4. Salvar metadata.json e frames_metadata.csv
            logger.info("--- [Passo 4/4] Gerando metadata.json e Registro Tabular CSV ---")
            calib_summary: Optional[Dict[str, Any]] = None
            if self.calibration_profile is not None:
                intr = self.calibration_profile.intrinsics
                he = self.calibration_profile.hand_eye
                val_m = he.get("validation_metrics", {})
                calib_summary = {
                    "calibration_id": self.calibration_profile.calibration_id,
                    "K": intr.get("K"),
                    "K_rectified": intr.get("K_rectified"),
                    "distortion_coeffs": intr.get("distortion_coeffs"),
                    "T_flange_camera": he.get("T_flange_to_camera"),
                    "T_tcp_camera": he.get("T_tcp_approx_to_camera"),
                    "intrinsic_reprojection_rmse_px": intr.get("reprojection_rmse_px"),
                    "hand_eye_translation_rmse_mm": val_m.get("board_translation_rmse_mm"),
                    "profile": self.calibration_profile.to_dict(),
                }

            full_metadata = {
                "dataset_version": self.config.dataset.dataset_version,
                "schema_version": self.config.dataset.schema_version,
                "session_id": self.config.dataset.session_id,
                "generated_at": datetime.now().isoformat(),
                "dry_run": self.config.dry_run,
                "plan_sha256": plan_doc.plan_sha256,
                "calibration": calib_summary,
                "provenance": {
                    "smartphone_id": self.config.smartphone.smartphone_id,
                    "camera_physical_id": self.config.smartphone.camera_physical_id,
                    "mount_id": self.config.robot.mount_id,
                    "tool_definition_id": self.config.robot.tool_definition_id,
                },
                "camera_info": {
                    "app": "Cam-App (com.robotics.camera - Camera2 HTTP Server)",
                    "active_presets": self.config.smartphone.active_presets,
                    "mode_descriptions": {
                        "auto": AUTO_MODE_DESCRIPTION,
                        "manual_calibrado": MANUAL_CALIBRADO_DESCRIPTION,
                    },
                    "auto_3a_settle_sec": self.config.smartphone.auto_3a_settle_sec,
                    "manual_settle_sec": self.config.smartphone.manual_settle_sec,
                    "devices": self.smartphone.device_names,
                },
                "robot_info": {
                    "model": "DENSO VS-050",
                    "ip_address": self.config.robot.ip_address,
                    "euler_convention": self.euler_convention,
                    "robot_settling_time_sec": self.config.robot.robot_settling_time_sec,
                    "mount_id": self.config.robot.mount_id,
                    "tool_definition_id": self.config.robot.tool_definition_id,
                    "tcp_offset": self.config.robot.tcp_offset.to_dict(),
                    "tcp_matrix_flange_to_tcp_4x4": tcp_mat_list,
                },
                "sampling_config": self.config.sampling.to_dict(),
                "scenes": metadata_scenes,
            }

            metadata_file = self.output_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(full_metadata, f, indent=2, ensure_ascii=False)

            if self.config.dataset.save_tabular_csv and tabular_rows:
                csv_file = self.output_dir / "frames_metadata.csv"
                fieldnames = list(tabular_rows[0].keys())
                with open(csv_file, "w", encoding="utf-8", newline="") as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(tabular_rows)
                logger.info(f"Registro tabular salvo em: {csv_file.resolve()} ({len(tabular_rows)} quadros)")

            console.rule("[bold green]Coleta Concluída com Sucesso![/bold green]")
            logger.info(f"Metadados completos salvos em: {metadata_file.resolve()}")
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
