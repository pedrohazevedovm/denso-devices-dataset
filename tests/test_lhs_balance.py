"""Testes estatísticos para auditoria de equilíbrio LHS e substituição Space-Filling na família P7."""

import json
import math
from pathlib import Path
import numpy as np

from src.visual_servoing_dataset.collector.dataset_collector import DatasetCollector
from src.visual_servoing_dataset.collector.session_manager import SessionManager
from src.visual_servoing_dataset.config import (
    AppConfig,
    BasePose,
    DatasetConfig,
    PoseSamplingConfig,
    RobotConfig,
    SceneConfig,
    SmartphoneConfig,
    TCPOffset,
)
from src.visual_servoing_dataset.robot.lhs_diagnostics import (
    compare_lhs_original_vs_final,
    evaluate_lhs_sample_set,
)
from src.visual_servoing_dataset.robot.pose_generator import PoseFamilyGenerator


def test_lhs_original_set_statistical_balance_n30():
    """Valida que o conjunto P7 original (N=30, d=6) atende a todos os critérios de equilíbrio estatístico."""
    sampling_cfg = PoseSamplingConfig(
        lhs_samples=30,
        lhs_shrinkage_factor=0.75,
        lhs_seed=42,
        lhs_optimization="random-cd",
    )
    gen = PoseFamilyGenerator(sampling_cfg, euler_convention="xyz")
    p0 = BasePose([218.54, -316.54, 552.22, -45.54, 87.49, -134.55, 5])
    lower = [-30.0, -25.0, -40.0, -10.0, -10.0, -15.0]
    upper = [30.0, 25.0, 40.0, 10.0, 10.0, 15.0]

    _, lhs_meta = gen.generate_scene_poses(
        scene_id="scene_01",
        p0=p0,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    u_orig = lhs_meta.normalized_samples_u
    assert len(u_orig) == 30

    metrics = evaluate_lhs_sample_set(u_orig)
    assert metrics["valid"] is True
    assert metrics["balanced"] is True
    assert metrics["discrepancy"]["cd2_percentile_vs_mc"] <= 5.0
    assert metrics["marginal_summary"]["min_stratum_coverage"] == 1.0
    assert metrics["marginal_summary"]["min_ks_pvalue"] > 0.50
    assert metrics["orthogonality"]["max_abs_pearson"] < 0.30
    assert metrics["space_filling"]["min_pairwise_dist"] > 0.35
    maxpro = metrics["space_filling"]["maxpro_criterion"]
    assert math.isfinite(maxpro) and maxpro > 0.0


def test_space_filling_replacement_preserves_balance_and_avoids_rejected():
    """Valida que 3 substituições sequenciais Space-Filling preservam equilíbrio e evitam regiões rejeitadas."""
    sampling_cfg = PoseSamplingConfig(
        lhs_samples=30,
        lhs_shrinkage_factor=0.75,
        lhs_seed=42,
        lhs_optimization="random-cd",
    )
    gen = PoseFamilyGenerator(sampling_cfg, euler_convention="xyz")
    p0 = BasePose([218.54, -316.54, 552.22, -45.54, 87.49, -134.55, 5])
    lower = [-30.0, -25.0, -40.0, -10.0, -10.0, -15.0]
    upper = [30.0, 25.0, 40.0, 10.0, 10.0, 15.0]

    _, lhs_meta = gen.generate_scene_poses(
        scene_id="scene_01",
        p0=p0,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    u_orig = [list(row) for row in lhs_meta.normalized_samples_u]
    u_arr = np.asarray(u_orig, dtype=np.float64)

    # Seleciona 1 amostra extrema + 2 outras amostras distintas
    extreme_idx = int(np.argmax(np.max(np.abs(u_arr - 0.5), axis=1)))
    reject_indices = [extreme_idx]
    for candidate_idx in (0, 14, 21):
        if candidate_idx not in reject_indices and len(reject_indices) < 3:
            reject_indices.append(candidate_idx)

    active_u = [list(row) for row in u_orig]
    rejected_u: list[list[float]] = []

    # Rejeita e substitui sequencialmente 3 vezes
    for rep_idx, orig_idx in enumerate(reject_indices, start=1):
        u_rej = u_orig[orig_idx]
        # Remove o ponto rejeitado do conjunto ativo atual
        pos_in_active = next(i for i, row in enumerate(active_u) if np.allclose(row, u_rej))
        active_u.pop(pos_in_active)
        rejected_u.append(u_rej)

        cand = gen.generate_lhs_replacement_candidate(
            scene_id="scene_01",
            base_pose=p0,
            rejected_pose_id=f"p7_lhs_{orig_idx + 1:03d}",
            replacement_index=rep_idx,
            lower_bounds=lower,
            upper_bounds=upper,
            active_u_samples=active_u,
            rejected_u_samples=rejected_u,
            target_n_samples=30,
        )
        assert cand.replacement_selection_metrics is not None
        assert cand.lhs_normalized_u is not None

        u_new = np.asarray(cand.lhs_normalized_u, dtype=np.float64)
        min_dist_to_rej = min(float(np.linalg.norm(u_new - np.asarray(r))) for r in rejected_u)
        assert min_dist_to_rej >= 0.10

        active_u.append(list(cand.lhs_normalized_u))

    assert len(active_u) == 30
    report = compare_lhs_original_vs_final(u_orig, active_u)
    assert report["has_replacements"] is True
    assert report["overall_balanced"] is True
    assert report["comparison"] is not None
    assert report["comparison"]["no_distribution_shift_ok"] is True
    assert report["comparison"]["min_dist_preserved_ok"] is True
    assert report["final_metrics"]["marginal_summary"]["min_stratum_coverage"] >= 0.90
    assert report["final_metrics"]["orthogonality"]["max_abs_pearson"] <= 0.30


def test_session_manager_persists_balance_report_and_analyze_plan(tmp_path: Path):
    """Valida integração em generate_plan -> verify_plan (rejeições) -> analyze_plan -> collect."""
    out_dir = tmp_path / "dataset_lhs"
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
            isolated_fractions=[-1.0, 1.0],
            lhs_samples=30,
            lhs_shrinkage_factor=0.75,
            lhs_seed=42,
            lhs_optimization="random-cd",
        ),
        scenes=[
            SceneConfig(
                name="scene_lhs",
                scene_category="planar_core",
                p0_pose=BasePose([440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1]),
                lower_bounds=[-30.0, -25.0, -40.0, -10.0, -10.0, -15.0],
                upper_bounds=[30.0, 25.0, 40.0, 10.0, 10.0, 15.0],
            )
        ],
        dataset=DatasetConfig(output_dir=str(out_dir)),
    )

    mgr = SessionManager(cfg)
    plan_file = mgr.generate_plan()
    assert plan_file.exists()

    frozen_doc = mgr.verify_plan(
        plan_path=plan_file,
        interactive=False,
        reject_pose_ids=["p7_lhs_003", "p7_lhs_018"],
        default_rejection_reason="Fora do ROI 80%",
    )
    assert frozen_doc.status == "frozen"

    with open(plan_file, "r", encoding="utf-8") as f:
        plan_json = json.load(f)

    plan_report = plan_json["scenes"]["scene_lhs"]["lhs_balance_report"]
    assert plan_report is not None
    assert plan_report["has_replacements"] is True
    assert plan_report["overall_balanced"] is True

    reports = mgr.analyze_plan(plan_path=plan_file)
    assert reports["scene_lhs"]["overall_balanced"] is True

    md_path = out_dir / "lhs_balance_report.md"
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "Resumo Global (Original vs. Conjunto Final)" in md_text
    assert "Análise Marginal por Eixo" in md_text
    assert "Matriz de Correlação de Pearson" in md_text

    collector = DatasetCollector(cfg, plan_path=plan_file)
    assert collector.run() is True

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_json = json.load(f)

    meta_report = meta_json["scenes"]["scene_lhs"]["lhs_balance_report"]
    assert meta_report is not None
    assert meta_report["has_replacements"] is True
    assert meta_report["overall_balanced"] is True


def test_evaluate_lhs_detects_unbalanced_biased_set():
    """Garante que um conjunto enviesado em um eixo e altamente correlacionado seja reprovado."""
    rng = np.random.default_rng(123)
    n = 30
    u = rng.uniform(0.0, 1.0, size=(n, 6))
    # Enviesa eixo 0 (x) para [0.0, 0.4] e acopla eixo 1 (y) fortemente a x
    u[:, 0] = rng.uniform(0.0, 0.4, size=n)
    u[:, 1] = np.clip(u[:, 0] + rng.normal(0.0, 0.02, size=n), 0.0, 0.999)

    metrics = evaluate_lhs_sample_set(u)
    assert metrics["valid"] is True
    assert metrics["balanced"] is False
    assert metrics["quality_checks"]["ks_uniformity_ok"] is False
    assert metrics["quality_checks"]["orthogonality_ok"] is False
    assert metrics["quality_checks"]["space_filling_cd2_ok"] is False


def test_reverify_and_reject_poses_on_frozen_plan(tmp_path: Path):
    """Valida reabertura controlada, rejeição e substituição rastreável sobre um poses_plan.json já congelado."""
    out_dir = tmp_path / "dataset_frozen_reverify"
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
            isolated_fractions=[-1.0, 1.0],
            lhs_samples=30,
            lhs_shrinkage_factor=0.75,
            lhs_seed=42,
            lhs_optimization="random-cd",
        ),
        scenes=[
            SceneConfig(
                name="scene_01",
                scene_category="black_cube",
                p0_pose=BasePose([-302.691, -129.623, 395.596, -178.1301, 48.1851, 55.5024, 5]),
                lower_bounds=[-30.0, -25.0, -40.0, -10.0, -10.0, -15.0],
                upper_bounds=[30.0, 25.0, 40.0, 10.0, 10.0, 15.0],
            ),
            SceneConfig(
                name="scene_02",
                scene_category="model_violation",
                p0_pose=BasePose([-38.61861, 257.5869, 477.2345, -178.1336, 48.18206, -39.37456, 5]),
                lower_bounds=[-30.0, -25.0, -40.0, -10.0, -10.0, -15.0],
                upper_bounds=[30.0, 25.0, 40.0, 10.0, 10.0, 15.0],
            ),
        ],
        dataset=DatasetConfig(output_dir=str(out_dir)),
    )

    mgr = SessionManager(cfg)
    plan_file = mgr.generate_plan()

    # Passo 1: Congelar pela primeira vez com 1 rejeição apenas na scene_01
    doc1 = mgr.verify_plan(
        plan_path=plan_file,
        interactive=False,
        reject_pose_ids=["scene_01:p7_lhs_001"],
    )
    assert doc1.status == "frozen"
    sha_v1 = doc1.plan_sha256
    assert sha_v1 is not None
    assert sum(1 for p in doc1.scenes["scene_01"].poses if p.replaces_pose_id is not None) == 1

    # Passo 2: Reabrir plano já congelado e rejeitar mais 2 poses apenas em scene_01
    doc2 = mgr.verify_plan(
        plan_path=plan_file,
        interactive=False,
        reject_pose_ids=["scene_01:p7_lhs_005", "scene_01:p7_lhs_012"],
        target_scenes=["scene_01"],
    )
    assert doc2.status == "frozen"
    assert doc2.plan_sha256 is not None and doc2.plan_sha256 != sha_v1

    s1_p7_rej = [p for p in doc2.scenes["scene_01"].poses if p.pose_family == "P7" and p.status == "rejected"]
    s1_p7_app = [p for p in doc2.scenes["scene_01"].poses if p.pose_family == "P7" and p.status == "approved"]
    s1_reps = [p for p in doc2.scenes["scene_01"].poses if p.replaces_pose_id is not None]

    assert [p.pose_id for p in s1_p7_rej] == ["p7_lhs_001", "p7_lhs_005", "p7_lhs_012"]
    assert len(s1_p7_app) == 30
    assert len(s1_reps) == 3
    assert [p.replacement_seed for p in s1_reps] == [1043, 1044, 1045]
    assert len({p.pose_id for p in s1_reps}) == 3
    assert doc2.scenes["scene_01"].lhs_balance_report is not None
    assert doc2.scenes["scene_01"].lhs_balance_report["overall_balanced"] is True

    # Garantir que scene_02 permaneceu intacta
    s2_p7_rej = [p for p in doc2.scenes["scene_02"].poses if p.pose_family == "P7" and p.status == "rejected"]
    s2_p7_app = [p for p in doc2.scenes["scene_02"].poses if p.pose_family == "P7" and p.status == "approved"]
    assert len(s2_p7_rej) == 0
    assert len(s2_p7_app) == 30

    # Passo 3: Reabrir pose específica sem rejeitar e recongelar
    doc3 = mgr.verify_plan(
        plan_path=plan_file,
        interactive=False,
        reverify_pose_ids=["scene_01:p7_lhs_008"],
        target_scenes=["scene_01"],
    )
    assert doc3.status == "frozen"
    assert sum(1 for p in doc3.scenes["scene_01"].poses if p.pose_family == "P7" and p.status == "approved") == 30

    # Passo 4: Reabrir todas as poses P7 aprovadas da scene_01 mantendo histórico das 3 rejeitadas
    doc4 = mgr.verify_plan(
        plan_path=plan_file,
        interactive=False,
        reverify_all=True,
        target_scenes=["scene_01"],
    )
    assert doc4.status == "frozen"
    assert sum(1 for p in doc4.scenes["scene_01"].poses if p.pose_family == "P7" and p.status == "rejected") == 3
    assert sum(1 for p in doc4.scenes["scene_01"].poses if p.pose_family == "P7" and p.status == "approved") == 30
