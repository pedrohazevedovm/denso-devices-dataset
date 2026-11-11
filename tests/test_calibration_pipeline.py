"""Testes unitários e de integração da Fase 0: Abóbada Esférica, Calibração 2 em 1 (K, D + Hand-Eye) e Homografia H_gt."""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.visual_servoing_dataset.calibration.calibration_generator import (
    CharucoBoardSpec,
    SphericalDomeGenerator,
    generate_charuco_board_files,
)
from src.visual_servoing_dataset.calibration.calibration_manager import CalibrationManager
from src.visual_servoing_dataset.calibration.calibration_solver import (
    CharucoCalibrationSolver,
    CharucoFrameDetection,
    ImageRectifier,
    compute_planar_homography_gt,
    compute_plane_normal_and_distance_in_cam,
)
from src.visual_servoing_dataset.collector.dataset_collector import DatasetCollector
from src.visual_servoing_dataset.config import (
    AppConfig,
    BasePose,
    CalibrationConfig,
    DatasetConfig,
    Pose,
    PoseSamplingConfig,
    RobotConfig,
    SceneConfig,
    SmartphoneConfig,
    TCPOffset,
)
from src.visual_servoing_dataset.robot.kinematics import invert_se3, pose_to_matrix


def test_spherical_dome_geometry_and_pivot_collinearity():
    """Valida as 24 poses da abóbada esférica em SE(3) e que o eixo +Z_T dos Blocos 1 e 2 intercepta o pivô exato."""
    p0 = Pose(440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1)
    d0_mm = 350.0
    dome_gen = SphericalDomeGenerator(d0_mm=d0_mm, euler_convention="xyz")

    t_b_t0 = pose_to_matrix(p0, euler_convention="xyz")
    t_b_pivot = dome_gen.compute_pivot_matrix(p0)

    # 1. Identidade quando alpha = beta = gamma = 0, dx = dy = 0 e d_k = d_0
    t_identity = dome_gen.compute_orbit_pose_matrix(t_b_pivot, 0.0, 0.0, 0.0, 0.0, 0.0, d0_mm)
    np.testing.assert_allclose(t_identity, t_b_t0, atol=1e-11)

    # 2. Geração das 24 poses (3 blocos de 8 poses)
    poses, pivot_out = dome_gen.generate_dome_poses(p0)
    assert len(poses) == 24
    np.testing.assert_allclose(pivot_out, t_b_pivot, atol=1e-12)

    block_counts = {1: 0, 2: 0, 3: 0}
    p_pivot = t_b_pivot[:3, 3]

    for dp in poses:
        block_counts[dp.block_id] += 1
        t_mat = np.asarray(dp.target_matrix_4x4, dtype=np.float64)
        assert t_mat.shape == (4, 4)

        # Ortonormalidade em SO(3) e det(R) = +1
        r_k = t_mat[:3, :3]
        np.testing.assert_allclose(r_k.T @ r_k, np.eye(3), atol=1e-7)
        assert np.isclose(np.linalg.det(r_k), 1.0, atol=1e-7)
        np.testing.assert_allclose(t_mat[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

        # Nos Blocos 1 e 2, o raio óptico +Z_T deve interceptar exatamente o centro do pivô ^B T_{pivot}
        if dp.block_id in (1, 2):
            p_k = t_mat[:3, 3]
            z_k = r_k[:, 2]  # Eixo +Z_T na base
            ray_to_pivot = p_pivot - p_k

            # O ponto p_k + d_k * z_k deve coincidir exatamente com p_pivot
            intersection_point = p_k + dp.radius_d_mm * z_k
            np.testing.assert_allclose(intersection_point, p_pivot, atol=1e-5)

            # Produto vetorial entre +Z_T e (p_pivot - p_k) deve ser zero (colinearidade estrita)
            cross_norm = float(np.linalg.norm(np.cross(z_k, ray_to_pivot)))
            assert cross_norm < 1e-5

    assert block_counts == {1: 8, 2: 8, 3: 8}


def test_synthetic_end_to_end_intrinsics_and_hand_eye_recovery():
    """Valida a recuperação exata de K, D, ^T T_C e ^B T_M pelo solver sobre as 24 poses da abóbada."""
    p0 = Pose(440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1)
    spec = CharucoBoardSpec(squares_x=7, squares_y=5, square_length_m=0.030, marker_length_m=0.022)
    dome_gen = SphericalDomeGenerator(d0_mm=350.0, euler_convention="xyz")
    solver = CharucoCalibrationSolver(spec, min_charuco_corners=8, hand_eye_method="PARK")

    dome_poses, t_b_pivot = dome_gen.generate_dome_poses(p0)

    # Ground Truth sintético
    resolution_px = (1920, 1080)
    k_gt = np.array(
        [[1450.2, 0.0, 960.5], [0.0, 1451.0, 540.2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    d_gt = np.array([-0.042, 0.115, -0.0002, 0.0004, -0.051], dtype=np.float64)

    r_tc_gt = Rotation.from_euler("xyz", [1.5, -1.1, 2.8], degrees=True).as_matrix()
    t_tc_gt_mm = np.array([3.2, -2.1, 5.4], dtype=np.float64)
    t_tcp_to_cam_gt = np.eye(4, dtype=np.float64)
    t_tcp_to_cam_gt[:3, :3] = r_tc_gt
    t_tcp_to_cam_gt[:3, 3] = t_tc_gt_mm

    board_shift = np.eye(4, dtype=np.float64)
    board_shift[:3, 3] = [-spec.board_width_mm / 2.0, -spec.board_height_mm / 2.0, 0.0]
    t_base_to_board_gt = t_b_pivot @ board_shift

    detections = []
    obj_pts_mm = solver.board_corners_3d_mm
    all_ids = np.arange(len(obj_pts_mm), dtype=np.int32)

    for dp in dome_poses:
        t_b_tk = np.asarray(dp.target_matrix_4x4, dtype=np.float64)
        t_b_ck = t_b_tk @ t_tcp_to_cam_gt
        t_ck_m = invert_se3(t_b_ck) @ t_base_to_board_gt

        rvec, _ = cv2.Rodrigues(t_ck_m[:3, :3])
        tvec = t_ck_m[:3, 3].reshape(3, 1)
        proj_2d, _ = cv2.projectPoints(
            obj_pts_mm.reshape(-1, 1, 3), rvec, tvec, k_gt, d_gt
        )
        pts_2d = proj_2d.reshape(-1, 2)

        detections.append(
            CharucoFrameDetection(
                pose_id=dp.pose_id,
                valid=True,
                corners_count=len(all_ids),
                charuco_corners_2d=pts_2d,
                charuco_ids=all_ids,
                object_points_3d_mm=obj_pts_mm,
                t_base_to_tcp_mm=t_b_tk,
            )
        )

    sol = solver.solve_from_detections(detections=detections, resolution_px=resolution_px)

    # 1. Erro intrínseco fx, fy, cx, cy < 1.0 px
    assert abs(sol.k_matrix[0, 0] - k_gt[0, 0]) < 1.0
    assert abs(sol.k_matrix[1, 1] - k_gt[1, 1]) < 1.0
    assert abs(sol.k_matrix[0, 2] - k_gt[0, 2]) < 1.0
    assert abs(sol.k_matrix[1, 2] - k_gt[1, 2]) < 1.0
    assert sol.reprojection_rmse_px < 0.05

    # 2. Erro Hand-Eye ^T t_C < 1.0 mm e ^T R_C < 0.2 deg
    trans_err_mm = float(np.linalg.norm(sol.t_tcp_to_cam_mm[:3, 3] - t_tc_gt_mm))
    rot_err_deg = float(
        Rotation.from_matrix(r_tc_gt.T @ sol.t_tcp_to_cam_mm[:3, :3]).magnitude() * (180.0 / np.pi)
    )
    assert trans_err_mm < 1.0, f"Erro de translação Hand-Eye excedeu 1.0 mm: {trans_err_mm}"
    assert rot_err_deg < 0.2, f"Erro de rotação Hand-Eye excedeu 0.2 deg: {rot_err_deg}"

    # 3. Pose do tabuleiro na base com board_translation_rmse_mm próximo de zero
    assert sol.board_translation_rmse_mm < 0.05
    assert sol.board_rotation_rmse_deg < 0.05
    np.testing.assert_allclose(sol.t_base_to_board_mean_mm[:3, 3], t_base_to_board_gt[:3, 3], atol=1.0)


def test_rectification_and_planar_homography_gt():
    """Valida a remoção de distorção pelo ImageRectifier e o mapeamento exato de H_gt (Corner Error < 1e-4 px)."""
    resolution_px = (1920, 1080)
    k_gt = np.array(
        [[1450.2, 0.0, 960.5], [0.0, 1451.0, 540.2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    d_gt = np.array([-0.042, 0.115, -0.0002, 0.0004, -0.051], dtype=np.float64)
    rectifier = ImageRectifier(k_matrix=k_gt, distortion_coeffs=d_gt, resolution_px=resolution_px)

    # 4 cantos 3D do plano do tabuleiro (Z = 0 no frame do tabuleiro M)
    corners_board_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [180.0, 0.0, 0.0],
            [180.0, 120.0, 0.0],
            [0.0, 120.0, 0.0],
        ],
        dtype=np.float64,
    )

    p0 = Pose(440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1)
    dome_gen = SphericalDomeGenerator(d0_mm=350.0, euler_convention="xyz")
    dome_poses, t_b_pivot = dome_gen.generate_dome_poses(p0)

    board_shift = np.eye(4, dtype=np.float64)
    board_shift[:3, 3] = [-90.0, -60.0, 0.0]
    t_b_board = t_b_pivot @ board_shift

    t_b_c1 = pose_to_matrix(p0, euler_convention="xyz")
    t_b_c2 = np.asarray(dome_poses[4].target_matrix_4x4, dtype=np.float64)  # Vista inclinada diagonal

    # Projetar cantos na vista C1 (com distorção D_gt e sem distorção pinhole puro)
    t_c1_m = invert_se3(t_b_c1) @ t_b_board
    rvec1, _ = cv2.Rodrigues(t_c1_m[:3, :3])
    tvec1 = t_c1_m[:3, 3].reshape(3, 1)
    pts_c1_dist, _ = cv2.projectPoints(corners_board_3d.reshape(-1, 1, 3), rvec1, tvec1, k_gt, d_gt)
    pts_c1_pinhole, _ = cv2.projectPoints(
        corners_board_3d.reshape(-1, 1, 3), rvec1, tvec1, k_gt, np.zeros(5)
    )

    # 1. Validar que ImageRectifier remove a distorção de volta para coordenadas pinhole puras
    pts_c1_rect = rectifier.rectify_points(pts_c1_dist.reshape(-1, 2))
    np.testing.assert_allclose(pts_c1_rect, pts_c1_pinhole.reshape(-1, 2), atol=1e-3)

    # 2. Validar que compute_planar_homography_gt mapeia os 4 cantos de C1 para C2 com erro < 1e-4 px
    t_c2_m = invert_se3(t_b_c2) @ t_b_board
    rvec2, _ = cv2.Rodrigues(t_c2_m[:3, :3])
    tvec2 = t_c2_m[:3, 3].reshape(3, 1)
    pts_c2_pinhole, _ = cv2.projectPoints(
        corners_board_3d.reshape(-1, 1, 3), rvec2, tvec2, k_gt, np.zeros(5)
    )
    pts_c2_pinhole = pts_c2_pinhole.reshape(-1, 2)

    t_c1_to_c2 = invert_se3(t_b_c2) @ t_b_c1
    n_1, d_1 = compute_plane_normal_and_distance_in_cam(t_b_c1, t_b_board)
    h_gt = compute_planar_homography_gt(k_gt, t_c1_to_c2, n_1, d_1)

    pts_c1_h = np.hstack([pts_c1_pinhole.reshape(-1, 2), np.ones((4, 1))])
    pts_c2_mapped_h = (h_gt @ pts_c1_h.T).T
    pts_c2_mapped = pts_c2_mapped_h[:, :2] / pts_c2_mapped_h[:, 2:3]

    corner_error = float(np.max(np.linalg.norm(pts_c2_mapped - pts_c2_pinhole, axis=1)))
    assert corner_error < 1e-4, f"Corner error excedeu 1e-4 px: {corner_error}"


def test_dry_run_charuco_and_calibration_integration_with_collect(tmp_path: Path):
    """Valida a execução --dry-run da Fase 0 (--generate-charuco e --calibrate-camera) e integração com --collect."""
    out_dir = tmp_path / "dataset"
    cal_dir = out_dir / "calibrations"

    # 1. Testar geração de prancha ChArUco
    spec = CharucoBoardSpec()
    img_p, spec_p = generate_charuco_board_files(
        spec=spec,
        output_image_path=tmp_path / "charuco_board.png",
        output_spec_path=tmp_path / "charuco_spec.json",
    )
    assert img_p.exists() and img_p.stat().st_size > 0
    assert spec_p.exists()

    # 2. Testar Fase 0 (--calibrate-camera --dry-run)
    cfg = AppConfig(
        dry_run=True,
        robot=RobotConfig(
            euler_convention="xyz",
            mount_id="MOUNT_01",
            tool_definition_id="TCP_CAM_01",
            tcp_offset=TCPOffset(x=10.0, y=-5.0, z=40.0, rx=0.0, ry=0.0, rz=0.0, tag="SMARTPHONE_TOOL"),
        ),
        smartphone=SmartphoneConfig(
            ips=["192.168.155.44"],
            camera_names={"192.168.155.44": "cam_01_moto_razr_60_ultra"},
        ),
        sampling=PoseSamplingConfig(
            isolated_fractions=[-1.0, 1.0],
            lhs_samples=2,
            lhs_shrinkage_factor=0.75,
            lhs_seed=42,
        ),
        calibration=CalibrationConfig(
            calibration_id="calib_test_001",
            output_dir=str(cal_dir),
            rectify_on_collect=True,
            d0_mm=350.0,
        ),
        scenes=[
            SceneConfig(
                name="scene_01",
                scene_category="planar_core",
                p0_pose=BasePose([440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1]),
            )
        ],
        dataset=DatasetConfig(output_dir=str(out_dir)),
    )

    cal_mgr = CalibrationManager(cfg)
    prof_doc, prof_path = cal_mgr.run_calibration()
    assert prof_path.exists()
    assert prof_doc.calibration_id == "calib_test_001"
    assert prof_doc.intrinsics["reprojection_rmse_px"] < 0.5
    assert prof_doc.hand_eye["validation_metrics"]["valid_poses_count"] == 24
    assert prof_doc.hand_eye["validation_metrics"]["board_translation_rmse_mm"] < 2.0

    # 3. Testar integração automática com Fase 3 (--collect --dry-run)
    collector = DatasetCollector(cfg, calibration_file=prof_path)
    assert collector.run() is True

    meta_file = out_dir / "metadata.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["calibration"] is not None
    assert meta["calibration"]["calibration_id"] == "calib_test_001"

    first_pair = meta["scenes"]["scene_01"]["pairs"][0]
    assert first_pair["robot_geometry"]["homography_gt_from_p0_3x3"] is not None
    cam_modes = first_pair["captures"]["cam_01_moto_razr_60_ultra"]
    rect_rel = cam_modes["manual_calibrado"]["rectified_relative_path"]
    assert rect_rel is not None
    assert (out_dir / rect_rel).exists()
