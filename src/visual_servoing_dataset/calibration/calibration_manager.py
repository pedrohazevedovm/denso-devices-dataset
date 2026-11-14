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
    matrix_to_pose,
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
    CharucoCenteringMeasurement,
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

    def auto_center_charuco(
        self,
        robot: Optional[DensoRobotController] = None,
        phone: Optional[SmartphoneController] = None,
        session_dir: Optional[Path] = None,
        initial_perturbation_se3: Optional[np.ndarray] = None,
    ) -> Tuple[Pose, List[Dict[str, Any]]]:
        """
        Centraliza automaticamente o tabuleiro ChArUco (via marcador central ID=8) e alinha o paralelismo
        entre o plano da câmera e o tabuleiro igualando a soma dos perímetros das extremidades
        (Esquerda: 15,16 vs Direita: 0,1 | Topo vs Base).
        """
        own_connections = robot is None or phone is None
        if own_connections:
            console.rule("[bold cyan]Fase 0-Pre: Centralização e Paralelismo Automático do ChArUco (ID=8 + Perímetros)[/bold cyan]")
            robot = DensoRobotController(self.config.robot, dry_run=self.config.dry_run)
            phone = SmartphoneController(self.config.smartphone, dry_run=self.config.dry_run)
            if not robot.connect():
                raise RuntimeError("Falha ao conectar com o robô DENSO VS-050 para centralização do ChArUco.")
            if not phone.connect():
                robot.disconnect()
                raise RuntimeError("Falha ao conectar com o Cam-App Android para centralização do ChArUco.")

        assert robot is not None and phone is not None
        out_dir = session_dir or (Path(self.cal_cfg.output_dir) / self.cal_cfg.calibration_id)
        center_dir = out_dir / "centering"
        center_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Posicionar inicialmente na pose nominal salva (p_cal_0_pose)
            p_init: Pose = self.cal_cfg.p_cal_0_pose.to_pose()
            if own_connections:
                logger.info(f"Movendo robô para pose nominal inicial P_cal,0: {p_init.to_list()}")
                robot.move_cartesian(p_init)

            current_pose = robot.get_cartesian_pose()
            primary_dev = phone.connected_devices[0]
            phone.apply_preset("auto", device_id=primary_dev)

            # No modo --dry-run, definir a pose fixa do tabuleiro no mundo a partir de p_init nominal
            t_b_init = pose_to_matrix(p_init, euler_convention=self.euler_convention)
            t_init_piv = np.eye(4, dtype=np.float64)
            t_init_piv[2, 3] = self.cal_cfg.d0_mm
            t_b_pivot_gt = t_b_init @ t_init_piv
            # Na montagem física em paisagem do suporte: +X_C = -Y_F, +Y_C = +X_F, +Z_C = +Z_F
            r_fc_nominal = np.array(
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            t_flange_to_cam_sim = np.eye(4, dtype=np.float64)
            t_flange_to_cam_sim[:3, :3] = r_fc_nominal
            board_shift = np.eye(4, dtype=np.float64)
            board_shift[:3, 3] = [
                -self.charuco_spec.board_width_mm / 2.0,
                -self.charuco_spec.board_height_mm / 2.0,
                0.0,
            ]
            # O tabuleiro no mundo (5x7 em paisagem) tem seus eixos alinhados com o flange em p_init
            t_base_to_board_sim = t_b_pivot_gt @ board_shift

            if self.config.dry_run and initial_perturbation_se3 is not None:
                t_pert_pose = matrix_to_pose(
                    t_b_init @ initial_perturbation_se3,
                    fig=p_init.fig,
                    euler_convention=self.euler_convention,
                )
                robot.move_cartesian(t_pert_pose)
                current_pose = robot.get_cartesian_pose()

            max_iters = max(1, int(self.cal_cfg.center_max_iterations))
            kp_trans = float(self.cal_cfg.center_kp_translation)
            kp_rot = float(self.cal_cfg.center_kp_rotation)
            max_step_mm = float(self.cal_cfg.center_max_step_mm)
            max_step_deg = float(self.cal_cfg.center_max_step_deg)

            history: List[Dict[str, Any]] = []

            for iteration in range(1, max_iters + 2):
                if not self.config.dry_run and self.cal_cfg.calibration_settling_time_sec > 0:
                    time.sleep(min(0.8, self.cal_cfg.calibration_settling_time_sec))

                current_pose = robot.get_cartesian_pose()
                t_b_curr = pose_to_matrix(current_pose, euler_convention=self.euler_convention)

                raw_img_path = center_dir / f"center_raw_iter_{iteration:02d}.jpg"
                dbg_img_path = center_dir / f"center_debug_iter_{iteration:02d}.jpg"

                if self.config.dry_run:
                    img_bgr = self._render_synthetic_charuco_image(
                        t_b_tk=t_b_curr,
                        t_flange_to_cam=t_flange_to_cam_sim,
                        t_base_to_board=t_base_to_board_sim,
                        resolution_px=(1920, 1080),
                    )
                    cv2.imwrite(str(raw_img_path), img_bgr)
                else:
                    saved = phone.capture_and_save_photo(
                        device_id=primary_dev,
                        target_dir=center_dir,
                        target_filename=raw_img_path.name,
                    )
                    if not saved or not saved.exists():
                        logger.warning(f"[Auto-Center] Falha ao capturar foto na iteração {iteration}.")
                        break
                    img_bgr = cv2.imread(str(saved), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        logger.warning(f"[Auto-Center] Imagem corrompida em {saved}.")
                        break

                meas = self.solver.analyze_charuco_centering_and_parallelism(
                    image_bgr=img_bgr,
                    center_marker_id=self.cal_cfg.center_marker_id,
                    debug_save_path=dbg_img_path,
                )

                if not meas.valid:
                    logger.warning(
                        f"[Auto-Center {iteration}] Detecção insuficiente ({meas.rejection_reason}). Mantendo pose atual."
                    )
                    break

                logger.info(
                    f"[Auto-Center {iteration}/{max_iters}] ID={meas.center_marker_id} em "
                    f"({meas.center_u_px:.1f}, {meas.center_v_px:.1f}) px | "
                    f"Erro Centro = {meas.center_error_norm_px:.1f} px ({meas.center_error_norm_mm:.2f} mm) | "
                    f"Perímetros Esq{meas.left_markers_used}={meas.sum_left_perimeters_px:.1f} px vs "
                    f"Dir{meas.right_markers_used}={meas.sum_right_perimeters_px:.1f} px (diff={meas.horiz_asymmetry_pct:+.2f}%) | "
                    f"Topo vs Base (diff={meas.vert_asymmetry_pct:+.2f}%) | Roll={meas.roll_error_deg:+.2f}°"
                )

                roll_ok = (not self.cal_cfg.center_align_roll) or (abs(meas.roll_error_deg) <= 1.0)
                converged = (
                    meas.center_error_norm_px <= self.cal_cfg.center_tolerance_px
                    and abs(meas.horiz_asymmetry_pct) <= self.cal_cfg.parallelism_tolerance_pct
                    and abs(meas.vert_asymmetry_pct) <= self.cal_cfg.parallelism_tolerance_pct
                    and roll_ok
                )

                step_record: Dict[str, Any] = {
                    "iteration": iteration,
                    "pose_cartesian": [round(float(v), 6) for v in current_pose.to_list()],
                    "measurement": meas.to_dict(),
                    "converged": converged,
                }
                history.append(step_record)

                if converged:
                    logger.info(
                        f"[Auto-Center Convergiu!] Tabuleiro centralizado e paralelo na iteração {iteration} "
                        f"(Erro Centro = {meas.center_error_norm_px:.1f} px / {meas.center_error_norm_mm:.2f} mm, "
                        f"Diff Perímetros Horiz = {meas.horiz_asymmetry_pct:+.2f}%, Vert = {meas.vert_asymmetry_pct:+.2f}%)."
                    )
                    break

                if iteration > max_iters:
                    logger.info(
                        f"[Auto-Center] Limite de {max_iters} iterações atingido "
                        f"(Erro residual = {meas.center_error_norm_px:.1f} px / {meas.center_error_norm_mm:.2f} mm)."
                    )
                    break

                # Cálculo do passo de controle no referencial da câmera C e mapeamento para o flange F via R_FC
                dt_cam_mm = np.array(
                    [kp_trans * meas.error_u_mm, kp_trans * meas.error_v_mm, 0.0],
                    dtype=np.float64,
                )
                d_omega_cam_deg = np.array(
                    [
                        +kp_rot * (meas.vert_asymmetry_pct / max(meas.vert_sensitivity_pct_per_deg, 0.3)),
                        -kp_rot * (meas.horiz_asymmetry_pct / 1.65),
                        +kp_rot * meas.roll_error_deg if self.cal_cfg.center_align_roll else 0.0,
                    ],
                    dtype=np.float64,
                )

                dt_flange_mm = r_fc_nominal @ dt_cam_mm
                d_omega_flange_deg = r_fc_nominal @ d_omega_cam_deg

                dx_f = float(np.clip(dt_flange_mm[0], -max_step_mm, max_step_mm))
                dy_f = float(np.clip(dt_flange_mm[1], -max_step_mm, max_step_mm))
                d_alpha_f = float(np.clip(d_omega_flange_deg[0], -max_step_deg, max_step_deg))
                d_beta_f = float(np.clip(d_omega_flange_deg[1], -max_step_deg, max_step_deg))
                d_gamma_f = float(np.clip(d_omega_flange_deg[2], -max_step_deg, max_step_deg))

                step_record["control_step_flange"] = {
                    "dx_mm": round(dx_f, 4),
                    "dy_mm": round(dy_f, 4),
                    "d_alpha_deg": round(d_alpha_f, 4),
                    "d_beta_deg": round(d_beta_f, 4),
                    "d_gamma_deg": round(d_gamma_f, 4),
                }

                # Atualizar pose em SE(3): translação paralela + rotação orbitando o pivô a d0_mm
                t_trans = np.eye(4, dtype=np.float64)
                t_trans[:3, 3] = [dx_f, dy_f, 0.0]
                t_piv = np.eye(4, dtype=np.float64)
                t_piv[2, 3] = self.cal_cfg.d0_mm
                t_unpiv = np.eye(4, dtype=np.float64)
                t_unpiv[2, 3] = -self.cal_cfg.d0_mm
                t_rot = np.eye(4, dtype=np.float64)
                t_rot[:3, :3] = Rotation.from_euler(
                    self.euler_convention, [d_alpha_f, d_beta_f, d_gamma_f], degrees=True
                ).as_matrix()

                t_b_next = t_b_curr @ t_trans @ t_piv @ t_rot @ t_unpiv
                next_pose = matrix_to_pose(
                    t_b_next, fig=current_pose.fig, euler_convention=self.euler_convention
                )

                if not robot.move_cartesian(next_pose):
                    logger.warning(
                        "[Auto-Center] Passo de correção excedeu alcance cinemático. Tentando apenas translação 2D..."
                    )
                    t_b_next_trans_only = t_b_curr @ t_trans
                    next_pose_trans = matrix_to_pose(
                        t_b_next_trans_only, fig=current_pose.fig, euler_convention=self.euler_convention
                    )
                    if not robot.move_cartesian(next_pose_trans):
                        logger.warning("[Auto-Center] Não foi possível avançar mais. Mantendo última pose válida.")
                        break

            final_pose = robot.get_cartesian_pose()
            log_path = center_dir / "centering_log.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "initial_pose_cartesian": [round(float(v), 6) for v in p_init.to_list()],
                        "final_centered_pose_cartesian": [round(float(v), 6) for v in final_pose.to_list()],
                        "iterations": history,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.info(
                f"Pose P_cal,0 refinada após centralização automática: "
                f"{[round(float(v), 3) for v in final_pose.to_list()]} (log salvo em {log_path})"
            )
            return final_pose, history

        finally:
            if own_connections:
                phone.close_cameras()
                robot.disconnect()

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
            # Passo 1B: Centralização e Paralelismo Automático em P_{cal,0} (ID=8 + Perímetros)
            # ------------------------------------------------------------------
            if self.cal_cfg.auto_center_p0 and not self.config.dry_run:
                logger.info("Executando centralização automática do ChArUco (ID=8) e paralelismo por perímetros...")
                measured_p0, _ = self.auto_center_charuco(
                    robot=robot,
                    phone=phone,
                    session_dir=calib_session_dir,
                )

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

    def _render_synthetic_charuco_image(
        self,
        t_b_tk: np.ndarray,
        t_flange_to_cam: np.ndarray,
        t_base_to_board: np.ndarray,
        resolution_px: Tuple[int, int] = (1920, 1080),
    ) -> np.ndarray:
        """Renderiza uma imagem perspectiva sintética real do tabuleiro ChArUco para simulação do Auto-Center."""
        width, height = resolution_px
        k_sim = np.array(
            [[1578.0, 0.0, width / 2.0], [0.0, 1600.0, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        t_b_ck = t_b_tk @ t_flange_to_cam
        t_ck_m = invert_se3(t_b_ck) @ t_base_to_board

        rvec, _ = cv2.Rodrigues(t_ck_m[:3, :3])
        tvec = t_ck_m[:3, 3].reshape(3, 1)

        bw_mm = self.charuco_spec.board_width_mm
        bh_mm = self.charuco_spec.board_height_mm
        tex_w = int(round(bw_mm * 6.0))
        tex_h = int(round(bh_mm * 6.0))
        board_gray = self.solver.board.generateImage((tex_w, tex_h), marginSize=0, borderBits=1)
        board_bgr = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)

        corners_3d = np.array(
            [[0.0, 0.0, 0.0], [bw_mm, 0.0, 0.0], [bw_mm, bh_mm, 0.0], [0.0, bh_mm, 0.0]],
            dtype=np.float64,
        ).reshape(-1, 1, 3)
        proj_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, k_sim, np.zeros(5, dtype=np.float64))
        dst_pts = proj_2d.reshape(4, 2).astype(np.float32)
        src_pts = np.array(
            [[0.0, 0.0], [float(tex_w), 0.0], [float(tex_w), float(tex_h)], [0.0, float(tex_h)]],
            dtype=np.float32,
        )
        h_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            board_bgr,
            h_persp,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(235, 235, 235),
        )
        return warped

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
