"""Testes unitários e de integração para cinemática SE(3), geração de poses P0..P7 (LHS) e protocolo em 3 fases."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.visual_servoing_dataset.collector.dataset_collector import DatasetCollector
from src.visual_servoing_dataset.collector.session_manager import SessionManager
from src.visual_servoing_dataset.config import (
    AppConfig,
    BasePose,
    DatasetConfig,
    Pose,
    PoseSamplingConfig,
    RobotConfig,
    SceneConfig,
    SmartphoneConfig,
    TCPOffset,
)
from src.visual_servoing_dataset.robot.kinematics import (
    apply_local_perturbation,
    compute_relative_transform,
    invert_se3,
    matrix_to_perturbation,
    matrix_to_pose,
    perturbation_to_matrix,
    pose_to_matrix,
)
from src.visual_servoing_dataset.robot.pose_generator import PoseFamilyGenerator
from src.visual_servoing_dataset.schemas import audit_camera_parameters


def test_se3_pose_matrix_round_trip():
    """Valida a conversão ida-e-volta entre Pose(X, Y, Z, Rx, Ry, Rz, Fig) do DENSO e matriz 4x4 em SE(3)."""
    original = Pose(x=440.25, y=-60.5, z=585.1, rx=16.0, ry=-22.5, rz=150.0, fig=5)
    mat = pose_to_matrix(original, euler_convention="xyz")

    assert mat.shape == (4, 4)
    # Verificar propriedades de SO(3): R^T * R = I e det(R) = +1
    r_mat = mat[:3, :3]
    np.testing.assert_allclose(r_mat.T @ r_mat, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r_mat), 1.0, atol=1e-12)
    np.testing.assert_allclose(mat[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    recovered = matrix_to_pose(mat, fig=original.fig, euler_convention="xyz")
    assert recovered.fig == original.fig
    np.testing.assert_allclose(recovered.to_list()[:6], original.to_list()[:6], atol=1e-9)


def test_se3_post_multiplication_and_relative_recovery():
    """Valida ^B T_{T,i} = ^B T_T(0) * Delta T_{T,i} e a recuperação exata de Delta T_{T,i}."""
    p0 = Pose(x=440.0, y=60.0, z=585.0, rx=16.0, ry=-22.0, rz=150.0, fig=1)
    xi_expected = [12.5, -18.0, -25.0, 4.5, -6.0, 11.2]

    target_pose, target_mat, delta_mat = apply_local_perturbation(p0, xi_expected, euler_convention="xyz")

    # Provar que pós-multiplicação em SE(3) NÃO é soma ingênua de coordenadas na base (pois R0 != I)
    naive_sum = [p0.x + xi_expected[0], p0.y + xi_expected[1], p0.z + xi_expected[2]]
    assert not np.allclose([target_pose.x, target_pose.y, target_pose.z], naive_sum, atol=1e-2)

    # Recuperar Delta T_{T,i} = (^B T_T(0))^{-1} * ^B T_{T,i}
    delta_recovered, xi_recovered = compute_relative_transform(p0, target_pose, euler_convention="xyz")
    np.testing.assert_allclose(delta_recovered, delta_mat, atol=1e-10)
    np.testing.assert_allclose(xi_recovered, xi_expected, atol=1e-9)

    # Verificar também inversão analítica SE(3)
    t_b_t0 = pose_to_matrix(p0, euler_convention="xyz")
    np.testing.assert_allclose(invert_se3(t_b_t0) @ t_b_t0, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(t_b_t0 @ delta_mat, target_mat, atol=1e-12)


def test_p0_to_p7_generation_and_lhs_stratification():
    """Valida a geração determinística de P0..P6 e a estratificação/limites das 30 amostras do LHS em P7."""
    lower = [-30.0, -25.0, -40.0, -10.0, -10.0, -15.0]
    upper = [30.0, 25.0, 40.0, 10.0, 10.0, 15.0]
    fractions = [-1.0, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.0]
    shrinkage = 0.75
    n_lhs = 30

    sampling_cfg = PoseSamplingConfig(
        lower_bounds=lower,
        upper_bounds=upper,
        isolated_fractions=fractions,
        lhs_samples=n_lhs,
        lhs_shrinkage_factor=shrinkage,
        lhs_seed=42,
        lhs_optimization="random-cd",
        capture_p0_start=True,
        capture_p0_end=True,
    )
    gen = PoseFamilyGenerator(sampling_cfg, euler_convention="xyz")
    p0 = Pose(440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1)

    candidates, lhs_meta = gen.generate_scene_poses("scene_01", p0)

    # Total esperado: 1 (P0 start) + 6 * 8 (P1..P6) + 30 (P7 LHS) + 1 (P0 end) = 80 poses
    assert len(candidates) == 1 + 6 * 8 + n_lhs + 1
    assert candidates[0].pose_id == "p0_reference_start"
    assert candidates[0].pose_family == "P0"
    assert candidates[-1].pose_id == "p0_reference_end"
    assert candidates[-1].pose_family == "P0"

    # Verificar que P1..P6 perturbam isoladamente apenas o eixo correspondente
    family_to_axis = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4, "P6": 5}
    for family, axis_idx in family_to_axis.items():
        fam_poses = [c for c in candidates if c.pose_family == family]
        assert len(fam_poses) == len(fractions)
        for c, f_expected in zip(fam_poses, fractions):
            for d in range(6):
                if d == axis_idx:
                    expected_val = abs(f_expected) * lower[d] if f_expected < 0 else f_expected * upper[d]
                    assert np.isclose(c.xi_local[d], expected_val, atol=1e-6)
                else:
                    assert np.isclose(c.xi_local[d], 0.0, atol=1e-9)

            # Verificar que a matriz 4x4 da pose-alvo recupera exatamente xi_local relativo a P0
            _, xi_rec = compute_relative_transform(p0, c.target_cartesian_pose, euler_convention="xyz")
            np.testing.assert_allclose(xi_rec, c.xi_local, atol=1e-5)

    # Verificar P7 (LHS): estratificação 1D em [0, 1)^6 e respeito estrito aos limites com shrinkage
    p7_poses = [c for c in candidates if c.pose_family == "P7"]
    assert len(p7_poses) == n_lhs

    u_norm = np.asarray(lhs_meta.normalized_samples_u, dtype=np.float64)
    xi_phys = np.asarray(lhs_meta.physical_samples_xi, dtype=np.float64)
    assert u_norm.shape == (n_lhs, 6)
    assert xi_phys.shape == (n_lhs, 6)

    # Cada dimensão d em [0..5] deve ter exatamente 1 amostra em cada estrato [k/N, (k+1)/N)
    for d in range(6):
        strata = np.floor(u_norm[:, d] * n_lhs).astype(int)
        assert sorted(strata.tolist()) == list(range(n_lhs))

    # Verificar limites físicos efetivos com shrinkage
    l_eff = shrinkage * np.asarray(lower)
    u_eff = shrinkage * np.asarray(upper)
    assert np.all(xi_phys >= l_eff - 1e-6)
    assert np.all(xi_phys <= u_eff + 1e-6)


def test_three_phase_protocol_and_rejection_replacement(tmp_path: Path):
    """Testa o fluxo completo das 3 fases (explore-limits -> generate-plan -> verify-plan com rejeição -> collect)."""
    cfg = AppConfig(
        dry_run=True,
        robot=RobotConfig(
            euler_convention="xyz",
            mount_id="MOUNT_TEST",
            tool_definition_id="TOOL_TEST",
            tcp_offset=TCPOffset(x=10.0, y=0.0, z=45.0, rx=0.0, ry=0.0, rz=0.0, tag="SMARTPHONE_TOOL"),
        ),
        smartphone=SmartphoneConfig(ips=["192.168.155.44"]),
        sampling=PoseSamplingConfig(
            isolated_fractions=[-1.0, 1.0],  # 2 frações por eixo (12 isoladas) + 4 LHS + 2 P0 = 18 poses
            lhs_samples=4,
            lhs_shrinkage_factor=0.75,
            lhs_seed=99,
        ),
        scenes=[
            SceneConfig(
                name="scene_test",
                scene_category="planar_core",
                p0_pose=BasePose([440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1]),
            )
        ],
        dataset=DatasetConfig(output_dir=str(tmp_path / "dataset_out")),
    )

    mgr = SessionManager(cfg)

    # Fase 1: Exploração de limites e calibração em P0
    session_plan_file = mgr.explore_limits(interactive=False)
    assert session_plan_file.exists()

    # Fase 2A: Geração do plano
    poses_plan_file = mgr.generate_plan(session_plan_path=session_plan_file)
    assert poses_plan_file.exists()

    # Fase 2B: Verificação supervisionada rejeitando 'p7_lhs_002'
    frozen_doc = mgr.verify_plan(
        plan_path=poses_plan_file,
        interactive=False,
        reject_pose_ids=["p7_lhs_002"],
        default_rejection_reason="Marcador próximo à borda da imagem",
    )
    assert frozen_doc.status == "frozen"
    assert frozen_doc.plan_sha256 is not None and len(frozen_doc.plan_sha256) == 64

    scene_plan = frozen_doc.scenes["scene_test"]
    rejected_list = [p for p in scene_plan.poses if p.status == "rejected"]
    assert len(rejected_list) == 1
    assert rejected_list[0].pose_id == "p7_lhs_002"
    assert rejected_list[0].rejection_reason == "Marcador próximo à borda da imagem"

    # Verificar que a substituta rastreável foi gerada e aprovada
    replacements = [p for p in scene_plan.poses if p.replaces_pose_id == "p7_lhs_002"]
    assert len(replacements) == 1
    assert replacements[0].status == "approved"
    assert replacements[0].replacement_seed == 99 + 1000 + 1

    # Fase 3: Coleta automática pareada a partir do plano congelado
    collector = DatasetCollector(cfg, plan_path=poses_plan_file)
    assert collector.run() is True

    meta_path = tmp_path / "dataset_out" / "metadata.json"
    csv_path = tmp_path / "dataset_out" / "frames_metadata.csv"
    assert meta_path.exists()
    assert csv_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["plan_sha256"] == frozen_doc.plan_sha256
    scene_meta = meta["scenes"]["scene_test"]
    assert len(scene_meta["rejected_poses_history"]) == 1
    assert len(scene_meta["pairs"]) == 18  # 1 P0_start + 12 P1..P6 + 4 P7 aprovadas + 1 P0_end
    assert scene_meta["p0_start_end_drift"]["start_to_end_trans_drift_mm"] == 0.0


def test_camera_parameter_audit():
    """Valida a detecção de divergências e campos nulos entre POST /config e X-Camera-Metadata."""
    requested = {
        "control_mode": "manual",
        "focus_distance": 1.8622,
        "iso": 1141,
        "exposure_time_ns": 16666660,
        "ois_enabled": False,
        "eis_enabled": False,
    }
    effective_ok = {
        "sensor_exposure_time": 16666660,
        "sensor_sensitivity": 1141,
        "sensor_timestamp": 123456789,
        "lens_focus_distance": 1.8622,
        "lens_state": 0,
        "control_ae_mode": 0,
        "control_af_mode": 0,
        "control_awb_mode": 0,
        "lens_optical_stabilization_mode": 0,
        "control_video_stabilization_mode": 0,
    }
    res_ok = audit_camera_parameters("manual_calibrado", requested, effective_ok)
    assert res_ok["compliant"] is True
    assert res_ok["divergences"] == {}

    effective_diverged = dict(effective_ok)
    effective_diverged["sensor_sensitivity"] = 800
    effective_diverged["lens_optical_stabilization_mode"] = 1
    res_bad = audit_camera_parameters("manual_calibrado", requested, effective_diverged)
    assert res_bad["compliant"] is False
    assert "iso" in res_bad["divergences"]
    assert "ois_enabled" in res_bad["divergences"]
