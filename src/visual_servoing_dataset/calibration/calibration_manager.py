"""Fase 0: Orquestrador da Calibração Automatizada '2 em 1' (Intrínsecos + Distorção + Hand-Eye via ChArUco)."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import AppConfig, Pose, ensure_pose
from ..robot.denso_controller import DensoRobotController
from ..robot.kinematics import (
    compute_pose_error,
    invert_se3,
    matrix_to_row_major_list,
    pose_to_matrix,
    tcp_offset_to_matrix,
)
from ..schemas import CalibrationProfileDocument
from ..smartphone.phone_controller import SmartphoneController
from ..utils.logger import console, logger
from .calibration_generator import (
    CalibrationDomePose,
    CharucoBoardSpec,
    SphericalDomeGenerator,
    build_charuco_board,
)
from .calibration_solver import (
    CharucoCalibrationSolver,
    CharucoFrameDetection,
    ImageRectifier,
)


class CalibrationManager:
    """Orquestra a coleta das 24 poses na Abóbada Esférica, o solver ChArUco + Hand-Eye e a retificação."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.cal_cfg = config.calibration
        self.euler_convention = config.robot.euler_convention

        self.charuco_spec = CharucoBoardSpec(
            squares_x=self.cal_cfg.charuco_squares_x,
            squares_y=self.cal_cfg.charuco_squares_y,
            square_length_m=self.cal_cfg.charuco_square_length_m,
            marker_length_m=self.cal_cfg.charuco_marker_length_m,
            dictionary=self.cal_cfg.charuco_dictionary,
        )
        self.dome_generator = SphericalDomeGenerator(
            d0_mm=self.cal_cfg.d0_mm,
            inner_tilt_deg=self.cal_cfg.inner_tilt_deg,
            outer_tilt_deg=self.cal_cfg.outer_tilt_deg,
            outer_rolls_deg=self.cal_cfg.outer_rolls_deg,
            outer_radii_factors=self.cal_cfg.outer_radii_factors,
            edge_tilt_deg=self.cal_cfg.edge_tilt_deg,
            edge_offset_fraction=self.cal_cfg.edge_offset_fraction,
            edge_radius_factor=self.cal_cfg.edge_radius_factor,
            euler_convention=self.euler_convention,
        )
        self.solver = CharucoCalibrationSolver(
            charuco_spec=self.charuco_spec,
            min_charuco_corners=self.cal_cfg.min_charuco_corners,
            hand_eye_method=self.cal_cfg.hand_eye_method,
            num_distortion_coeffs=self.cal_cfg.num_distortion_coeffs,
        )

    def run_calibration(
        self,
        verify_poses: bool = False,
        output_profile_path: Optional[Path] = None,
        synthetic_noise_px: float = 0.15,
    ) -> Tuple[CalibrationProfileDocument, Path]:
        """Executa a Fase 0 completa (24 poses na abóbada esférica em modo manual_calibrado + Solver 2 em 1)."""
        console.rule("[bold magenta]Fase 0: Calibração Automatizada '2 em 1' (Intrínsecos + Hand-Eye ChArUco)[/bold magenta]")

        calib_session_dir = Path(self.cal_cfg.output_dir) / self.cal_cfg.calibration_id
        images_dir = calib_session_dir / "images"
        debug_dir = calib_session_dir / "debug_detections"
        images_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        robot = DensoRobotController(self.config.robot, dry_run=self.config.dry_run)
        phone = SmartphoneController(self.config.smartphone, dry_run=self.config.dry_run)

        if not robot.connect():
            raise RuntimeError("Falha ao conectar com o robô DENSO VS-050 na Fase 0.")
        if not phone.connect():
            robot.disconnect()
            raise RuntimeError("Falha ao conectar com o Cam-App Android na Fase 0.")

        try:
            # ------------------------------------------------------------------
            # Passo 1: Posicionar/ler pose inicial central P_{cal,0} sobre o ChArUco
            # ------------------------------------------------------------------
            p_cal_0: Pose = self.cal_cfg.p_cal_0_pose.to_pose()
            logger.info(f"Posicionando robô na pose central de calibração P_cal,0 (d0={self.cal_cfg.d0_mm} mm)...")
            robot.move_cartesian(p_cal_0)
            measured_p0 = robot.get_cartesian_pose()

            # ------------------------------------------------------------------
            # Passo 2: Disparar captura 'auto' em P_{cal,0} e travar 'manual_calibrado'
            # ------------------------------------------------------------------
            logger.info("Calibrando foco, ISO e exposição a partir do modo 'auto' em P_cal,0 e travando 'manual_calibrado'...")
            calibrated_map = phone.calibrate_manual_from_auto(force=True)
            primary_dev = phone.connected_devices[0]
            phone.apply_preset("manual_calibrado", device_id=primary_dev)
            locked_cfg = calibrated_map.get(
                primary_dev,
                phone.get_device_presets(primary_dev).get("manual_calibrado", {}),
            )

            # ------------------------------------------------------------------
            # Passo 3: Gerar as 24 poses da Abóbada Esférica Centrada no Alvo
            # ------------------------------------------------------------------
            dome_poses, t_b_pivot = self.dome_generator.generate_dome_poses(measured_p0)
            logger.info(f"Geradas {len(dome_poses)} poses na Abóbada Esférica (3 blocos de 8 poses).")

            # ------------------------------------------------------------------
            # Passo 4: Coleta das 24 poses estritamente em 'manual_calibrado'
            # ------------------------------------------------------------------
            detections: List[CharucoFrameDetection] = []
            pose_records: List[Dict[str, Any]] = []
            resolution_px = (1920, 1080)
            interactive_verify = bool(verify_poses and not self.config.dry_run)

            # Ground truth sintético para validação matemática em modo --dry-run
            rng = np.random.default_rng(42)
            k_sim_gt = np.array(
                [[1450.0, 0.0, 960.0], [0.0, 1452.0, 540.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dist_sim_gt = np.array([-0.042, 0.115, -0.0002, 0.0004, -0.051], dtype=np.float64)
            r_tcp_cam_sim = Rotation.from_euler("xyz", [1.2, -0.8, 2.5], degrees=True).as_matrix()
            t_tcp_cam_sim_mm = np.array([2.5, -1.5, 4.0], dtype=np.float64)
            t_tcp_to_cam_sim_gt = np.eye(4, dtype=np.float64)
            t_tcp_to_cam_sim_gt[:3, :3] = r_tcp_cam_sim
            t_tcp_to_cam_sim_gt[:3, 3] = t_tcp_cam_sim_mm

            # Posicionar a origem do tabuleiro de forma que o centro do ChArUco coincida com o pivô ^B T_{pivot}
            board_center_shift = np.eye(4, dtype=np.float64)
            board_center_shift[:3, 3] = [
                -self.charuco_spec.board_width_mm / 2.0,
                -self.charuco_spec.board_height_mm / 2.0,
                0.0,
            ]
            t_base_to_board_sim_gt = t_b_pivot @ board_center_shift

            for dp in dome_poses:
                if interactive_verify:
                    ans = console.input(
                        f"[Verificação] Mover e capturar {dp.pose_id} (Bloco {dp.block_id}: "
                        f"alpha={dp.alpha_pitch_deg}, beta={dp.beta_yaw_deg}, gamma={dp.gamma_roll_deg}, r={dp.radius_d_mm}mm)? "
                        f"[Enter=Sim / p=Pular esta pose / s=Seguir automático nas demais]: "
                    ).strip().lower()
                    if ans == "s":
                        interactive_verify = False
                    elif ans == "p":
                        logger.info(f"Pose {dp.pose_id} pulada manualmente pelo operador.")
                        continue

                logger.info(
                    f"-> [{dp.pose_index:02d}/24] {dp.pose_id} ({dp.block_name}) | "
                    f"Tilt=({dp.alpha_pitch_deg:+.1f}°, {dp.beta_yaw_deg:+.1f}°, {dp.gamma_roll_deg:+.1f}°) | "
                    f"Shift=({dp.dx_mm:+.1f}, {dp.dy_mm:+.1f}) mm | d={dp.radius_d_mm:.1f} mm"
                )
                moved = robot.move_cartesian(dp.target_cartesian_pose)
                if not moved:
                    logger.warning(
                        f"Não foi possível alcançar {dp.pose_id} (fora de alcance ou erro do controlador). "
                        f"Pulando captura desta pose para evitar foto duplicada."
                    )
                    pose_records.append({
                        **dp.to_dict(),
                        "pose_measured_before": None,
                        "pose_measured_after": None,
                        "inter_capture_drift_mm": None,
                        "inter_capture_drift_deg": None,
                        "charuco_valid": False,
                        "charuco_corners_count": 0,
                        "rejection_reason": "robot_out_of_range",
                    })
                    continue

                if not self.config.dry_run and self.cal_cfg.calibration_settling_time_sec > 0:
                    time.sleep(self.cal_cfg.calibration_settling_time_sec)

                pose_before = robot.get_cartesian_pose()
                t_b_tk = pose_to_matrix(pose_before, euler_convention=self.euler_convention)

                img_path = images_dir / dp.filename
                debug_path = debug_dir / f"debug_{dp.filename}"

                if self.config.dry_run:
                    det = self._simulate_dry_run_frame(
                        dp=dp,
                        t_b_tk=t_b_tk,
                        t_tcp_to_cam_gt=t_tcp_to_cam_sim_gt,
                        t_base_to_board_gt=t_base_to_board_sim_gt,
                        k_gt=k_sim_gt,
                        dist_gt=dist_sim_gt,
                        resolution_px=resolution_px,
                        noise_std_px=synthetic_noise_px,
                        rng=rng,
                        save_img_path=img_path,
                        debug_img_path=debug_path,
                    )
                else:
                    saved = phone.capture_and_save_photo(
                        device_id=primary_dev,
                        target_dir=images_dir,
                        target_filename=dp.filename,
                    )
                    if not saved or not saved.exists():
                        logger.warning(f"Falha ao capturar imagem em {dp.pose_id}.")
                        continue
                    img_bgr = cv2.imread(str(saved), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        logger.warning(f"Imagem corrompida em {saved}.")
                        continue
                    resolution_px = (int(img_bgr.shape[1]), int(img_bgr.shape[0]))
                    det = self.solver.detect_charuco_in_image(
                        image_bgr=img_bgr,
                        pose_id=dp.pose_id,
                        t_base_to_tcp_mm=t_b_tk,
                        debug_save_path=debug_path,
                    )
                    self.charuco_spec = self.solver.charuco_spec
                    if det.valid:
                        logger.info(
                            f"[ChArUco OK] {dp.pose_id}: {det.corners_count} cantos internos detectados "
                            f"(debug salvo em {debug_path.name})."
                        )
                    else:
                        logger.warning(
                            f"[ChArUco Rejeitado] {dp.pose_id}: {det.corners_count} cantos detectados "
                            f"({det.rejection_reason})."
                        )

                pose_after = robot.get_cartesian_pose()
                drift_mm, drift_deg = compute_pose_error(
                    pose_before, pose_after, euler_convention=self.euler_convention
                )

                detections.append(det)
                pose_records.append({
                    **dp.to_dict(),
                    "pose_measured_before": [round(float(v), 6) for v in pose_before.to_list()],
                    "pose_measured_after": [round(float(v), 6) for v in pose_after.to_list()],
                    "inter_capture_drift_mm": round(drift_mm, 6),
                    "inter_capture_drift_deg": round(drift_deg, 6),
                    "charuco_valid": det.valid,
                    "charuco_corners_count": det.corners_count,
                    "rejection_reason": det.rejection_reason,
                })

            # Retornar à pose central P_cal,0 ao finalizar a varredura
            robot.move_cartesian(measured_p0)

            # ------------------------------------------------------------------
            # Passo 5: Executar o Solucionador "2 em 1" (Intrínsecos + Hand-Eye)
            # ------------------------------------------------------------------
            t_flange_to_tcp_mm = tcp_offset_to_matrix(
                self.config.robot.tcp_offset, euler_convention=self.euler_convention
            )
            cam_meta = (
                phone.get_last_telemetry(primary_dev)
                if self.cal_cfg.use_camera2_initial_guess
                else None
            )

            solution = self.solver.solve_from_detections(
                detections=detections,
                resolution_px=resolution_px,
                t_flange_to_tcp_mm=t_flange_to_tcp_mm,
                camera2_metadata=cam_meta,
            )

            profile_doc = CalibrationProfileDocument(
                schema_version=self.config.dataset.schema_version,
                calibration_id=self.cal_cfg.calibration_id,
                timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                smartphone_id=self.config.smartphone.smartphone_id,
                camera_physical_id=self.config.smartphone.camera_physical_id,
                mount_id=self.config.robot.mount_id,
                tool_definition_id=self.config.robot.tool_definition_id,
                resolution_px=[int(resolution_px[0]), int(resolution_px[1])],
                locked_camera_params={
                    "focus_distance_diopters": round(float(locked_cfg.get("focus_distance", 1.8622)), 6),
                    "iso": int(locked_cfg.get("iso", 1141)),
                    "exposure_time_ns": int(locked_cfg.get("exposure_time_ns", 16666660)),
                    "ois": "off",
                    "eis": "off",
                },
                charuco_spec=self.charuco_spec.to_dict(),
                intrinsics={
                    "K": matrix_to_row_major_list(solution.k_matrix),
                    "K_rectified": matrix_to_row_major_list(solution.k_rectified),
                    "distortion_model": solution.distortion_model,
                    "distortion_coeffs": [round(float(c), 8) for c in solution.distortion_coeffs],
                    "camera2_initial_guess_used": solution.camera2_initial_guess_used,
                    "reprojection_rmse_px": solution.reprojection_rmse_px,
                    "per_view_reprojection_rmse_px": solution.per_view_reprojection_rmse_px,
                },
                hand_eye={
                    "method": solution.hand_eye_method,
                    "T_tcp_approx_to_camera": matrix_to_row_major_list(solution.t_tcp_to_cam_m),
                    "T_flange_to_camera": matrix_to_row_major_list(solution.t_flange_to_cam_m),
                    "T_base_to_board_mean": matrix_to_row_major_list(solution.t_base_to_board_mean_m),
                    "T_tcp_approx_to_camera_mm": matrix_to_row_major_list(solution.t_tcp_to_cam_mm),
                    "T_flange_to_camera_mm": matrix_to_row_major_list(solution.t_flange_to_cam_mm),
                    "T_base_to_board_mean_mm": matrix_to_row_major_list(solution.t_base_to_board_mean_mm),
                    "plane_equation_in_base": {
                        "normal": solution.plane_normal_in_base,
                        "distance_m": solution.plane_distance_in_base_m,
                    },
                    "validation_metrics": {
                        "valid_poses_count": solution.valid_poses_count,
                        "board_translation_rmse_mm": solution.board_translation_rmse_mm,
                        "board_translation_std_mm": solution.board_translation_std_mm,
                        "board_rotation_rmse_deg": solution.board_rotation_rmse_deg,
                        "alternative_methods_comparison": solution.alternative_methods_comparison,
                    },
                },
            )

            # Salvar calibration_profile.json na pasta da calibração e no caminho configurado
            session_profile_path = calib_session_dir / "calibration_profile.json"
            with open(session_profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_doc.to_dict(), f, indent=2, ensure_ascii=False)

            with open(calib_session_dir / "dome_poses_log.json", "w", encoding="utf-8") as f:
                json.dump(pose_records, f, indent=2, ensure_ascii=False)

            final_profile_path = (
                Path(output_profile_path)
                if output_profile_path
                else (
                    Path(self.cal_cfg.calibration_file)
                    if self.cal_cfg.calibration_file
                    else (Path(self.config.dataset.output_dir) / "calibration_profile.json")
                )
            )
            final_profile_path.parent.mkdir(parents=True, exist_ok=True)
            if final_profile_path.resolve() != session_profile_path.resolve():
                with open(final_profile_path, "w", encoding="utf-8") as f:
                    json.dump(profile_doc.to_dict(), f, indent=2, ensure_ascii=False)

            console.rule("[bold green]Relatório da Calibração '2 em 1' (Fase 0)[/bold green]")
            logger.info(f"Vistas válidas aprovadas: {solution.valid_poses_count}/24")
            logger.info(f"RMSE de Reprojeção Intrínseca: {solution.reprojection_rmse_px:.4f} px")
            logger.info(
                f"fx={solution.k_matrix[0, 0]:.2f}, fy={solution.k_matrix[1, 1]:.2f}, "
                f"cx={solution.k_matrix[0, 2]:.2f}, cy={solution.k_matrix[1, 2]:.2f}"
            )
            logger.info(f"Distorção ({solution.distortion_model}): {[round(float(x), 5) for x in solution.distortion_coeffs]}")
            logger.info(
                f"Consistência Hand-Eye ({solution.hand_eye_method}): "
                f"RMSE Translação = {solution.board_translation_rmse_mm:.4f} mm | "
                f"RMSE Rotação = {solution.board_rotation_rmse_deg:.4f}°"
            )
            logger.info(f"Perfil de calibração salvo em: {final_profile_path.resolve()}")

            return profile_doc, final_profile_path

        finally:
            phone.close_cameras()
            robot.disconnect()

    def _simulate_dry_run_frame(
        self,
        dp: CalibrationDomePose,
        t_b_tk: np.ndarray,
        t_tcp_to_cam_gt: np.ndarray,
        t_base_to_board_gt: np.ndarray,
        k_gt: np.ndarray,
        dist_gt: np.ndarray,
        resolution_px: Tuple[int, int],
        noise_std_px: float,
        rng: np.random.Generator,
        save_img_path: Path,
        debug_img_path: Path,
    ) -> CharucoFrameDetection:
        """Projeta sinteticamente os cantos 3D do ChArUco usando K_gt, D_gt e ^T T_{C,gt} para o modo --dry-run."""
        width, height = resolution_px
        t_b_ck = t_b_tk @ t_tcp_to_cam_gt
        t_ck_m = invert_se3(t_b_ck) @ t_base_to_board_gt

        rvec, _ = cv2.Rodrigues(t_ck_m[:3, :3])
        tvec = t_ck_m[:3, 3].reshape(3, 1)

        all_obj_pts_mm = self.solver.board_corners_3d_mm
        proj_pts, _ = cv2.projectPoints(
            all_obj_pts_mm.reshape(-1, 1, 3),
            rvec,
            tvec,
            k_gt,
            dist_gt,
        )
        pts_2d = proj_pts.reshape(-1, 2)
        if noise_std_px > 0.0:
            pts_2d = pts_2d + rng.normal(0.0, noise_std_px, size=pts_2d.shape)

        in_bounds = (
            (pts_2d[:, 0] >= 8.0)
            & (pts_2d[:, 0] < width - 8.0)
            & (pts_2d[:, 1] >= 8.0)
            & (pts_2d[:, 1] < height - 8.0)
        )
        valid_ids = np.where(in_bounds)[0].astype(np.int32)
        valid_pts_2d = pts_2d[valid_ids]
        valid_obj_mm = all_obj_pts_mm[valid_ids]

        # Renderizar imagem sintética com os cantos projetados para inspeção visual
        canvas = np.full((height, width, 3), (32, 36, 44), dtype=np.uint8)
        for pt, cid in zip(valid_pts_2d, valid_ids):
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(canvas, (px, py), 5, (0, 255, 128), -1)
            cv2.putText(canvas, str(cid), (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        cv2.imwrite(str(save_img_path), canvas)
        cv2.imwrite(str(debug_img_path), canvas)

        return CharucoFrameDetection(
            pose_id=dp.pose_id,
            valid=len(valid_ids) >= self.cal_cfg.min_charuco_corners,
            corners_count=int(len(valid_ids)),
            charuco_corners_2d=valid_pts_2d,
            charuco_ids=valid_ids,
            object_points_3d_mm=valid_obj_mm,
            t_base_to_tcp_mm=t_b_tk,
        )

    @staticmethod
    def rectify_dataset_directory(
        dataset_dir: Path,
        calibration_profile_path: Path,
    ) -> int:
        """Retifica todas as imagens brutas de um dataset existente gerando a árvore espelhada em rectified/."""
        if not calibration_profile_path.exists():
            raise FileNotFoundError(f"Arquivo de calibração não encontrado: {calibration_profile_path}")

        with open(calibration_profile_path, "r", encoding="utf-8") as f:
            prof_doc = CalibrationProfileDocument.from_dict(json.load(f))

        rectifier = ImageRectifier(
            k_matrix=prof_doc.intrinsics["K"],
            distortion_coeffs=prof_doc.intrinsics["distortion_coeffs"],
            resolution_px=(prof_doc.resolution_px[0], prof_doc.resolution_px[1]),
            k_rectified=prof_doc.intrinsics.get("K_rectified"),
        )

        rectified_count = 0
        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name in ("calibrations", "_test_captures"):
                continue
            for cam_dir in sorted(scene_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue
                for mode_dir in sorted(cam_dir.iterdir()):
                    if not mode_dir.is_dir() or mode_dir.name == "rectified":
                        continue
                    for img_file in sorted(mode_dir.glob("*.jpg")):
                        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        rect_img = rectifier.rectify_image(img)
                        out_path = cam_dir / "rectified" / mode_dir.name / img_file.name
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_path), rect_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        rectified_count += 1

        logger.info(f"Total de imagens retificadas salvas em '{dataset_dir}': {rectified_count}")
        return rectified_count
