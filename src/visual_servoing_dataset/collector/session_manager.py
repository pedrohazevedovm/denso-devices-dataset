"""Gerenciador das Fases 1 e 2 do Protocolo Experimental em 3 Fases.

- Fase 1 (--explore-limits): Definição de P0, calibração do modo 'manual_calibrado' a partir de 'auto'
  em P0 e exploração assistida dos limites [l, u] em SE(3) salvando session_plan.json.
- Fase 2 (--generate-plan e --verify-plan): Geração das famílias P0..P7 (incluindo LHS 6-DoF em P7),
  verificação supervisionada com substituição rastreável de amostras rejeitadas e congelamento SHA-256
  em poses_plan.json.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, Pose, ensure_pose
from ..robot.denso_controller import DensoRobotController
from ..robot.kinematics import (
    apply_local_perturbation,
    compute_relative_transform,
    matrix_to_row_major_list,
    pose_to_matrix,
    tcp_offset_to_matrix,
)
from ..robot.pose_generator import PoseFamilyGenerator
from ..schemas import PosesPlanDocument, ScenePosesPlan
from ..smartphone.phone_controller import SmartphoneController
from ..utils.logger import console, logger


AXIS_EXPLORATION_STEPS = [
    (0, +1, "+x", "Translação positiva em X_T0 (mm)"),
    (0, -1, "-x", "Translação negativa em X_T0 (mm)"),
    (1, +1, "+y", "Translação positiva em Y_T0 (mm)"),
    (1, -1, "-y", "Translação negativa em Y_T0 (mm)"),
    (2, +1, "+z", "Translação positiva em Z_T0 (mm)"),
    (2, -1, "-z", "Translação negativa em Z_T0 (mm)"),
    (3, +1, "+rx", "Rotação positiva em torno de X_T0 / Roll (graus)"),
    (3, -1, "-rx", "Rotação negativa em torno de X_T0 / Roll (graus)"),
    (4, +1, "+ry", "Rotação positiva em torno de Y_T0 / Pitch (graus)"),
    (4, -1, "-ry", "Rotação negativa em torno de Y_T0 / Pitch (graus)"),
    (5, +1, "+rz", "Rotação positiva em torno de Z_T0 / Yaw (graus)"),
    (5, -1, "-rz", "Rotação negativa em torno de Z_T0 / Yaw (graus)"),
]


class SessionManager:
    """Orquestra a Fase 1 (Exploração de Limites) e a Fase 2 (Geração e Verificação do Plano de Poses)."""

    def __init__(self, config: AppConfig, calibration_file: Optional[Path] = None):
        self.config = config
        self.euler_convention = config.robot.euler_convention
        self.calibration_file = calibration_file or (
            Path(config.calibration.calibration_file)
            if config.calibration.calibration_file
            else None
        )
        self.pose_generator = PoseFamilyGenerator(
            config.sampling,
            euler_convention=self.euler_convention,
        )

    # ------------------------------------------------------------------
    # FASE 1: Exploração de Limites e Definição de P0 (--explore-limits)
    # ------------------------------------------------------------------
    def explore_limits(
        self,
        output_path: Optional[Path] = None,
        interactive: bool = False,
    ) -> Path:
        """Executa a Fase 1: lê/posiciona P0, trava 'manual_calibrado' a partir do 'auto' e salva session_plan.json."""
        console.rule("[bold cyan]Fase 1: Exploração de Limites e Calibração em P0[/bold cyan]")
        plan_path = output_path or (Path(self.config.dataset.output_dir) / self.config.dataset.session_plan_file)
        plan_path.parent.mkdir(parents=True, exist_ok=True)

        robot = DensoRobotController(self.config.robot, dry_run=self.config.dry_run)
        phone = SmartphoneController(self.config.smartphone, dry_run=self.config.dry_run)

        if not robot.connect():
            raise RuntimeError("Falha ao conectar ao robô DENSO VS-050 na Fase 1.")
        if not phone.connect():
            robot.disconnect()
            raise RuntimeError("Falha ao conectar ao Cam-App Android na Fase 1.")

        scenes_session_data: Dict[str, Any] = {}

        try:
            for scene_cfg in self.config.scenes:
                logger.info(f"\n--- Explorando limites para cena: {scene_cfg.name} ({scene_cfg.scene_category}) ---")
                p0_pose: Pose = scene_cfg.p0_pose.to_pose()

                # Posicionar robô em P0 e registrar ^B T_T(0) e Joint(0)
                robot.move_cartesian(p0_pose)
                measured_p0 = robot.get_cartesian_pose()
                measured_j0 = robot.get_joints_pose()
                t_b_t0 = pose_to_matrix(measured_p0, euler_convention=self.euler_convention)

                # Disparar captura 'auto' em P0 e travar parâmetros em 'manual_calibrado'
                calibrated_map = phone.calibrate_manual_from_auto(force=True)
                if self.calibration_file and self.calibration_file.exists():
                    with open(self.calibration_file, "r", encoding="utf-8") as cf:
                        cal_doc = json.load(cf)
                    locked_focus = cal_doc.get("locked_camera_params", {}).get("focus_distance_diopters")
                    if locked_focus is not None:
                        for dev, cfg in calibrated_map.items():
                            auto_focus = float(cfg.get("focus_distance", locked_focus))
                            if abs(auto_focus - float(locked_focus)) > 0.05:
                                logger.warning(
                                    f"[AUDITORIA FOCO] Foco lido em P0 ({auto_focus:.4f} D) difere do foco da "
                                    f"calibração ({locked_focus:.4f} D). Travando em {locked_focus:.4f} D para preservar K!"
                                )
                            cfg["focus_distance"] = float(locked_focus)
                            phone.get_device_presets(dev)["manual_calibrado"]["focus_distance"] = float(locked_focus)

                calibrated_by_name = {
                    phone.device_names.get(dev, dev): cfg
                    for dev, cfg in calibrated_map.items()
                }

                # Determinar limites inferiores l e superiores u
                default_l = list(scene_cfg.lower_bounds or self.config.sampling.lower_bounds)
                default_u = list(scene_cfg.upper_bounds or self.config.sampling.upper_bounds)
                l_vec = list(default_l)
                u_vec = list(default_u)

                for axis_idx, sign, label, desc in AXIS_EXPLORATION_STEPS:
                    default_val = u_vec[axis_idx] if sign > 0 else l_vec[axis_idx]
                    xi_probe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    xi_probe[axis_idx] = default_val
                    probe_pose, _, _ = apply_local_perturbation(
                        measured_p0, xi_probe, euler_convention=self.euler_convention
                    )

                    if interactive and not self.config.dry_run:
                        user_in = console.input(
                            f"[bold yellow][{scene_cfg.name}] Limite {label} ({desc})[/bold yellow] "
                            f"- Digite valor, 'read' para ler pose atual do robô, ou Enter para manter [{default_val}]: "
                        ).strip()
                        if user_in.lower() == "read":
                            curr_pose = robot.get_cartesian_pose()
                            _, xi_meas = compute_relative_transform(
                                measured_p0, curr_pose, euler_convention=self.euler_convention
                            )
                            measured_val = round(float(xi_meas[axis_idx]), 4)
                        elif user_in:
                            measured_val = float(user_in)
                        else:
                            measured_val = float(default_val)
                    else:
                        # Simular deslocamento até o limite e recuperar via transformação relativa em SE(3)
                        robot.move_cartesian(probe_pose)
                        curr_pose = robot.get_cartesian_pose()
                        _, xi_meas = compute_relative_transform(
                            measured_p0, curr_pose, euler_convention=self.euler_convention
                        )
                        measured_val = round(float(xi_meas[axis_idx]), 4)

                    if sign > 0:
                        u_vec[axis_idx] = measured_val
                    else:
                        l_vec[axis_idx] = measured_val

                # Retornar a P0 ao concluir exploração da cena
                robot.move_cartesian(measured_p0)

                scenes_session_data[scene_cfg.name] = {
                    "scene_id": scene_cfg.name,
                    "scene_category": scene_cfg.scene_category,
                    "description": scene_cfg.description,
                    "p0_cartesian": [round(float(v), 6) for v in measured_p0.to_list()],
                    "p0_joints": [round(float(v), 6) for v in measured_j0.to_list()],
                    "p0_matrix_4x4": matrix_to_row_major_list(t_b_t0),
                    "lower_bounds": l_vec,
                    "upper_bounds": u_vec,
                    "calibrated_manual_settings": calibrated_by_name,
                }
                logger.info(
                    f"[{scene_cfg.name}] Limites registrados:\n"
                    f"  l = {l_vec}\n"
                    f"  u = {u_vec}"
                )

            tcp_mat = tcp_offset_to_matrix(self.config.robot.tcp_offset, euler_convention=self.euler_convention)
            session_plan = {
                "schema_version": self.config.dataset.schema_version,
                "session_id": self.config.dataset.session_id,
                "explored_at": datetime.now().isoformat(),
                "dry_run": self.config.dry_run,
                "euler_convention": self.euler_convention,
                "mount_id": self.config.robot.mount_id,
                "tool_definition_id": self.config.robot.tool_definition_id,
                "tcp_offset": self.config.robot.tcp_offset.to_dict(),
                "tcp_matrix_flange_to_tcp_4x4": matrix_to_row_major_list(tcp_mat),
                "scenes": scenes_session_data,
            }

            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(session_plan, f, indent=2, ensure_ascii=False)

            logger.info(f"Plano de sessão (Fase 1) salvo em: {plan_path.resolve()}")
            return plan_path

        finally:
            phone.close_cameras()
            robot.disconnect()

    # ------------------------------------------------------------------
    # FASE 2A: Geração de Poses Candidatas P0..P7 (--generate-plan)
    # ------------------------------------------------------------------
    def generate_plan(
        self,
        session_plan_path: Optional[Path] = None,
        output_plan_path: Optional[Path] = None,
    ) -> Path:
        """Gera todas as poses candidatas P0..P7 (com LHS em P7) e salva em poses_plan.json (pending_approval)."""
        console.rule("[bold cyan]Fase 2A: Geração do Plano de Poses (P0..P7 em SE(3) + LHS)[/bold cyan]")
        out_dir = Path(self.config.dataset.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        resolved_session_path = session_plan_path or (out_dir / self.config.dataset.session_plan_file)
        resolved_output_path = output_plan_path or (out_dir / self.config.dataset.poses_plan_file)

        session_scenes: Dict[str, Any] = {}
        if resolved_session_path.exists():
            logger.info(f"Carregando limites explorados na Fase 1 de: {resolved_session_path.resolve()}")
            with open(resolved_session_path, "r", encoding="utf-8") as f:
                session_doc = json.load(f)
                session_scenes = session_doc.get("scenes", {})

        scenes_plan_map: Dict[str, ScenePosesPlan] = {}

        for scene_cfg in self.config.scenes:
            s_info = session_scenes.get(scene_cfg.name, {})
            p0_list = s_info.get("p0_cartesian", scene_cfg.p0_pose.cartesian)
            p0_joints = s_info.get("p0_joints", scene_cfg.p0_pose.joints)
            p0_pose = ensure_pose(p0_list)
            p0_mat = pose_to_matrix(p0_pose, euler_convention=self.euler_convention)

            l_bounds = s_info.get(
                "lower_bounds",
                scene_cfg.lower_bounds or self.config.sampling.lower_bounds,
            )
            u_bounds = s_info.get(
                "upper_bounds",
                scene_cfg.upper_bounds or self.config.sampling.upper_bounds,
            )
            calibrated_manual = s_info.get("calibrated_manual_settings", {})

            all_candidates = []
            lhs_meta = None
            for rep_idx in range(max(1, self.config.sampling.repeat_count)):
                candidates, meta = self.pose_generator.generate_scene_poses(
                    scene_id=scene_cfg.name,
                    p0=p0_pose,
                    lower_bounds=l_bounds,
                    upper_bounds=u_bounds,
                    repeat_index=rep_idx,
                )
                all_candidates.extend(candidates)
                if lhs_meta is None:
                    lhs_meta = meta

            assert lhs_meta is not None
            scene_plan = ScenePosesPlan(
                scene_id=scene_cfg.name,
                scene_category=scene_cfg.scene_category,
                description=scene_cfg.description,
                euler_convention=self.euler_convention,
                p0_cartesian=[round(float(v), 6) for v in p0_pose.to_list()],
                p0_joints=p0_joints,
                p0_matrix_4x4=matrix_to_row_major_list(p0_mat),
                lower_bounds=[float(v) for v in l_bounds],
                upper_bounds=[float(v) for v in u_bounds],
                isolated_fractions=list(self.config.sampling.isolated_fractions),
                calibrated_manual_settings=calibrated_manual,
                lhs_metadata=lhs_meta,
                poses=all_candidates,
            )
            scenes_plan_map[scene_cfg.name] = scene_plan
            logger.info(
                f"[{scene_cfg.name}] Geradas {len(all_candidates)} poses candidatas "
                f"(P0 + P1..P6 isoladas + {lhs_meta.n_samples} amostras LHS em P7)."
            )

        tcp_mat = tcp_offset_to_matrix(self.config.robot.tcp_offset, euler_convention=self.euler_convention)
        plan_doc = PosesPlanDocument(
            schema_version=self.config.dataset.schema_version,
            session_id=self.config.dataset.session_id,
            generated_at=datetime.now().isoformat(),
            status="pending_approval",
            euler_convention=self.euler_convention,
            mount_id=self.config.robot.mount_id,
            tool_definition_id=self.config.robot.tool_definition_id,
            tcp_offset=self.config.robot.tcp_offset.to_dict(),
            tcp_matrix_flange_to_tcp_4x4=matrix_to_row_major_list(tcp_mat),
            scenes=scenes_plan_map,
        )

        with open(resolved_output_path, "w", encoding="utf-8") as f:
            json.dump(plan_doc.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Plano de poses (status='pending_approval') salvo em: {resolved_output_path.resolve()}")
        return resolved_output_path

    # ------------------------------------------------------------------
    # FASE 2B: Verificação Supervisionada e Congelamento (--verify-plan)
    # ------------------------------------------------------------------
    def verify_plan(
        self,
        plan_path: Optional[Path] = None,
        interactive: bool = False,
        reject_pose_ids: Optional[List[str]] = None,
        default_rejection_reason: str = "Objeto fora do campo de visão (FOV) ou oclusão",
    ) -> PosesPlanDocument:
        """Percorre e valida as poses geradas (com suporte a rejeição rastreável em P7) e congela o plano com SHA-256."""
        console.rule("[bold cyan]Fase 2B: Verificação Supervisionada e Congelamento do Plano[/bold cyan]")
        out_dir = Path(self.config.dataset.output_dir)
        resolved_plan_path = plan_path or (out_dir / self.config.dataset.poses_plan_file)

        if not resolved_plan_path.exists():
            logger.info(f"Arquivo '{resolved_plan_path}' ainda não existe. Gerando plano automaticamente...")
            self.generate_plan(output_plan_path=resolved_plan_path)

        with open(resolved_plan_path, "r", encoding="utf-8") as f:
            raw_doc = json.load(f)

        plan_doc = PosesPlanDocument.from_dict(raw_doc)
        reject_set = set(reject_pose_ids or [])

        robot = DensoRobotController(self.config.robot, dry_run=self.config.dry_run)
        robot.connect()

        try:
            for scene_id, scene_plan in plan_doc.scenes.items():
                logger.info(f"\n--- Verificando cena '{scene_id}' ({len(scene_plan.poses)} poses iniciais) ---")
                replacement_counter = 0
                idx = 0

                while idx < len(scene_plan.poses):
                    pose_cand = scene_plan.poses[idx]
                    if pose_cand.status in ("approved", "rejected"):
                        idx += 1
                        continue

                    # Mover robô na simulação ou hardware durante verificação de P7 (ou todas)
                    if pose_cand.pose_family == "P7" or not self.config.dry_run:
                        robot.move_cartesian(pose_cand.target_cartesian_pose)

                    should_reject = False
                    reason = default_rejection_reason

                    if pose_cand.pose_id in reject_set:
                        should_reject = True
                    elif interactive and not self.config.dry_run and pose_cand.pose_family == "P7":
                        ans = console.input(
                            f"Pose [{pose_cand.pose_id}] xi={pose_cand.xi_local} -> [A]provar (Enter) / [R]ejeitar? "
                        ).strip().lower()
                        if ans.startswith("r"):
                            should_reject = True
                            custom_reason = console.input("Motivo da rejeição: ").strip()
                            if custom_reason:
                                reason = custom_reason

                    now_iso = datetime.now().isoformat()
                    if should_reject:
                        # NÃO apagar nem alterar silenciosamente: marcar rejected e gerar substituta rastreável
                        pose_cand.status = "rejected"
                        pose_cand.rejection_reason = reason
                        pose_cand.rejected_at = now_iso
                        pose_cand.verified_at = now_iso
                        logger.warning(
                            f"[{scene_id}] Pose '{pose_cand.pose_id}' REJEITADA (motivo: {reason}). "
                            f"Registro preservado com status='rejected'."
                        )

                        if pose_cand.pose_family == "P7":
                            replacement_counter += 1
                            rep_cand = self.pose_generator.generate_lhs_replacement_candidate(
                                scene_id=scene_id,
                                base_pose=scene_plan.p0_cartesian,
                                rejected_pose_id=pose_cand.pose_id,
                                replacement_index=replacement_counter,
                                lower_bounds=scene_plan.lower_bounds,
                                upper_bounds=scene_plan.upper_bounds,
                                repeat_index=pose_cand.repeat_index,
                            )
                            logger.info(
                                f"[{scene_id}] Amostra substituta gerada: '{rep_cand.pose_id}' "
                                f"(replaces='{rep_cand.replaces_pose_id}', seed={rep_cand.replacement_seed})"
                            )
                            # Inserir logo antes de p0_reference_end (ou ao final)
                            if scene_plan.poses and scene_plan.poses[-1].variation_type == "reference_end":
                                scene_plan.poses.insert(len(scene_plan.poses) - 1, rep_cand)
                            else:
                                scene_plan.poses.append(rep_cand)
                    else:
                        pose_cand.status = "approved"
                        pose_cand.verified_at = now_iso

                    idx += 1

            sha256_hash = plan_doc.freeze(datetime.now().isoformat())
            with open(resolved_plan_path, "w", encoding="utf-8") as f:
                json.dump(plan_doc.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(
                f"Plano de poses CONGELADO (status='frozen', SHA-256={sha256_hash}) em: {resolved_plan_path.resolve()}"
            )
            return plan_doc

        finally:
            robot.disconnect()
