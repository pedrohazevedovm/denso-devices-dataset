"""Gerador das famílias de poses P0..P7 em SE(3) com Latin Hypercube Sampling (LHS)."""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy
from scipy.stats import qmc

from ..config import BasePose, Pose, PoseSamplingConfig, ensure_pose
from ..schemas import LHSGenerationMetadata, PlannedPoseCandidate
from .kinematics import (
    apply_local_perturbation,
    matrix_to_row_major_list,
    pose_to_matrix,
)


ISOLATED_FAMILY_SPECS: List[Tuple[str, int, str, str]] = [
    ("P1", 0, "tx", "trans_x"),
    ("P2", 1, "ty", "trans_y"),
    ("P3", 2, "tz", "trans_z"),
    ("P4", 3, "rx", "rot_x_roll"),
    ("P5", 4, "ry", "rot_y_pitch"),
    ("P6", 5, "rz", "rot_z_yaw"),
]


class PoseFamilyGenerator:
    """Gera as famílias de poses P0 a P7 por pós-multiplicação em SE(3) no referencial local T_0."""

    def __init__(
        self,
        sampling_config: PoseSamplingConfig,
        euler_convention: str = "xyz",
    ):
        self.config = sampling_config
        self.euler_convention = euler_convention

    def compute_effective_lhs_bounds(
        self,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        shrinkage_factor: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula os limites efetivos [l_eff, u_eff] aplicando o fator de redução (shrinkage) em torno de P0 (0)."""
        l_raw = np.asarray(lower_bounds if lower_bounds is not None else self.config.lower_bounds, dtype=np.float64)
        u_raw = np.asarray(upper_bounds if upper_bounds is not None else self.config.upper_bounds, dtype=np.float64)
        alpha = float(shrinkage_factor if shrinkage_factor is not None else self.config.lhs_shrinkage_factor)
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"lhs_shrinkage_factor deve estar em (0, 1], recebido: {alpha}")
        if np.any(l_raw > u_raw):
            raise ValueError(f"lower_bounds ({l_raw}) não pode exceder upper_bounds ({u_raw}).")

        l_eff = alpha * l_raw
        u_eff = alpha * u_raw
        return l_eff, u_eff

    def generate_scene_poses(
        self,
        scene_id: str,
        p0: Union[BasePose, Pose, Sequence[float]],
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        repeat_index: int = 0,
    ) -> Tuple[List[PlannedPoseCandidate], LHSGenerationMetadata]:
        """Gera todas as poses candidatas P0_start, P1..P6, P7 (LHS) e P0_end para uma cena."""
        base_pose: Pose = p0.to_pose() if isinstance(p0, BasePose) else ensure_pose(p0)
        l_vec = [float(v) for v in (lower_bounds if lower_bounds is not None else self.config.lower_bounds)]
        u_vec = [float(v) for v in (upper_bounds if upper_bounds is not None else self.config.upper_bounds)]

        candidates: List[PlannedPoseCandidate] = []

        # 1. P0 inicial (p0_reference_start)
        if self.config.capture_p0_start:
            candidates.append(
                self._build_candidate(
                    scene_id=scene_id,
                    pose_id="p0_reference_start",
                    pose_family="P0",
                    variation_type="reference_start",
                    base_pose=base_pose,
                    xi=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    repeat_index=repeat_index,
                )
            )

        # 2. Famílias isoladas determinísticas P1 a P6
        for family_code, axis_idx, axis_short, var_type in ISOLATED_FAMILY_SPECS:
            for frac in self.config.isolated_fractions:
                f_val = float(frac)
                if abs(f_val) < 1e-9:
                    continue

                xi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                if f_val < 0.0:
                    xi[axis_idx] = abs(f_val) * l_vec[axis_idx] if l_vec[axis_idx] <= 0.0 else f_val * abs(l_vec[axis_idx])
                    sign_tag = f"m{int(round(abs(f_val) * 100)):03d}"
                else:
                    xi[axis_idx] = f_val * u_vec[axis_idx]
                    sign_tag = f"p{int(round(f_val * 100)):03d}"

                pose_id = f"{family_code.lower()}_{axis_short}_{sign_tag}"
                candidates.append(
                    self._build_candidate(
                        scene_id=scene_id,
                        pose_id=pose_id,
                        pose_family=family_code,
                        variation_type=var_type,
                        base_pose=base_pose,
                        xi=xi,
                        repeat_index=repeat_index,
                        axis_fraction=f_val,
                    )
                )

        # 3. Família combinada 6-DoF via Latin Hypercube Sampling (P7)
        lhs_candidates, lhs_meta = self._generate_p7_lhs_candidates(
            scene_id=scene_id,
            base_pose=base_pose,
            l_vec=l_vec,
            u_vec=u_vec,
            repeat_index=repeat_index,
        )
        candidates.extend(lhs_candidates)

        # 4. P0 final (p0_reference_end) para verificação de deriva mecânica e iluminação
        if self.config.capture_p0_end:
            candidates.append(
                self._build_candidate(
                    scene_id=scene_id,
                    pose_id="p0_reference_end",
                    pose_family="P0",
                    variation_type="reference_end",
                    base_pose=base_pose,
                    xi=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    repeat_index=repeat_index,
                )
            )

        return candidates, lhs_meta

    def _generate_p7_lhs_candidates(
        self,
        scene_id: str,
        base_pose: Pose,
        l_vec: List[float],
        u_vec: List[float],
        repeat_index: int,
    ) -> Tuple[List[PlannedPoseCandidate], LHSGenerationMetadata]:
        l_eff, u_eff = self.compute_effective_lhs_bounds(l_vec, u_vec, self.config.lhs_shrinkage_factor)
        n_samples = int(self.config.lhs_samples)

        if n_samples > 0:
            sampler = qmc.LatinHypercube(
                d=6,
                optimization=self.config.lhs_optimization,  # type: ignore[arg-type]
                seed=self.config.lhs_seed,
            )
            u_norm = sampler.random(n=n_samples)
            xi_matrix = l_eff + u_norm * (u_eff - l_eff)
        else:
            u_norm = np.zeros((0, 6), dtype=np.float64)
            xi_matrix = np.zeros((0, 6), dtype=np.float64)

        p7_candidates: List[PlannedPoseCandidate] = []
        for idx in range(n_samples):
            u_row = [round(float(v), 8) for v in u_norm[idx]]
            xi_row = [round(float(v), 6) for v in xi_matrix[idx]]
            sample_num = idx + 1
            pose_id = f"p7_lhs_{sample_num:03d}"
            p7_candidates.append(
                self._build_candidate(
                    scene_id=scene_id,
                    pose_id=pose_id,
                    pose_family="P7",
                    variation_type="lhs_6dof",
                    base_pose=base_pose,
                    xi=xi_row,
                    repeat_index=repeat_index,
                    lhs_sample_index=sample_num,
                    lhs_normalized_u=u_row,
                )
            )

        lhs_meta = LHSGenerationMetadata(
            seed=self.config.lhs_seed,
            scipy_version=scipy.__version__,
            optimization=self.config.lhs_optimization,
            n_samples=n_samples,
            raw_lower_bounds=list(l_vec),
            raw_upper_bounds=list(u_vec),
            shrinkage_factor=self.config.lhs_shrinkage_factor,
            effective_lower_bounds=[round(float(v), 6) for v in l_eff],
            effective_upper_bounds=[round(float(v), 6) for v in u_eff],
            normalized_samples_u=[[round(float(v), 8) for v in row] for row in u_norm],
            physical_samples_xi=[[round(float(v), 6) for v in row] for row in xi_matrix],
        )
        return p7_candidates, lhs_meta

    def generate_lhs_replacement_candidate(
        self,
        scene_id: str,
        base_pose: Union[BasePose, Pose, Sequence[float]],
        rejected_pose_id: str,
        replacement_index: int,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        repeat_index: int = 0,
    ) -> PlannedPoseCandidate:
        """Gera uma amostra substituta rastreável para uma pose P7 rejeitada durante --verify-plan."""
        p0_obj: Pose = base_pose.to_pose() if isinstance(base_pose, BasePose) else ensure_pose(base_pose)
        l_eff, u_eff = self.compute_effective_lhs_bounds(lower_bounds, upper_bounds, self.config.lhs_shrinkage_factor)
        derived_seed = int(self.config.lhs_seed) + 1000 + int(replacement_index)

        sampler = qmc.LatinHypercube(
            d=6,
            optimization=self.config.lhs_optimization,  # type: ignore[arg-type]
            seed=derived_seed,
        )
        u_norm = sampler.random(n=1)[0]
        xi_phys = l_eff + u_norm * (u_eff - l_eff)

        u_list = [round(float(v), 8) for v in u_norm]
        xi_list = [round(float(v), 6) for v in xi_phys]
        new_pose_id = f"{rejected_pose_id}_rep{replacement_index:02d}"

        candidate = self._build_candidate(
            scene_id=scene_id,
            pose_id=new_pose_id,
            pose_family="P7",
            variation_type="lhs_6dof_replacement",
            base_pose=p0_obj,
            xi=xi_list,
            repeat_index=repeat_index,
            lhs_sample_index=replacement_index,
            lhs_normalized_u=u_list,
        )
        candidate.replaces_pose_id = rejected_pose_id
        candidate.replacement_seed = derived_seed
        return candidate

    def _build_candidate(
        self,
        scene_id: str,
        pose_id: str,
        pose_family: str,
        variation_type: str,
        base_pose: Pose,
        xi: Sequence[float],
        repeat_index: int = 0,
        axis_fraction: Optional[float] = None,
        lhs_sample_index: Optional[int] = None,
        lhs_normalized_u: Optional[List[float]] = None,
    ) -> PlannedPoseCandidate:
        target_pose, target_mat, delta_mat = apply_local_perturbation(
            base_pose, xi, euler_convention=self.euler_convention
        )
        pair_id = f"{scene_id}__{pose_id}__r{repeat_index:02d}"
        filename = f"{pose_id}.jpg"
        return PlannedPoseCandidate(
            pose_id=pose_id,
            pair_id=pair_id,
            pose_family=pose_family,
            variation_type=variation_type,
            filename=filename,
            repeat_index=repeat_index,
            xi_local=[round(float(v), 6) for v in xi],
            delta_t_local_4x4=matrix_to_row_major_list(delta_mat),
            target_cartesian_pose=[round(float(v), 6) for v in target_pose.to_list()],
            target_matrix_4x4=matrix_to_row_major_list(target_mat),
            status="pending_approval",
            axis_fraction=axis_fraction,
            lhs_sample_index=lhs_sample_index,
            lhs_normalized_u=lhs_normalized_u,
        )
