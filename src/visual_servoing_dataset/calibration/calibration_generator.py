"""Fase 0: Gerador da prancha ChArUco e das 24 poses da Abóbada Esférica Centrada no Alvo em SE(3)."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import BasePose, Pose, ensure_pose
from ..robot.kinematics import matrix_to_pose, matrix_to_row_major_list, pose_to_matrix


ARUCO_DICT_MAP: Dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}


@dataclass
class CharucoBoardSpec:
    """Especificação geométrica do tabuleiro ChArUco."""
    squares_x: int = 7
    squares_y: int = 5
    square_length_m: float = 0.030
    marker_length_m: float = 0.022
    dictionary: str = "DICT_4X4_50"

    @property
    def board_width_mm(self) -> float:
        return round(self.squares_x * self.square_length_m * 1000.0, 4)

    @property
    def board_height_mm(self) -> float:
        return round(self.squares_y * self.square_length_m * 1000.0, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "dictionary": self.dictionary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharucoBoardSpec":
        return cls(
            squares_x=int(data.get("squares_x", 7)),
            squares_y=int(data.get("squares_y", 5)),
            square_length_m=float(data.get("square_length_m", 0.030)),
            marker_length_m=float(data.get("marker_length_m", 0.022)),
            dictionary=str(data.get("dictionary", "DICT_4X4_50")),
        )


@dataclass
class CalibrationDomePose:
    """Representa uma das 24 poses da Abóbada Esférica Centrada no Alvo."""
    pose_index: int                        # 1..24
    pose_id: str                           # ex: "cal_b1_01", "cal_b2_09", "cal_b3_17"
    block_id: int                          # 1, 2 ou 3
    block_name: str                        # "inner_ring_focal", "outer_ring_hand_eye", "edge_corner_distortion"
    filename: str                          # ex: "cal_b1_01.jpg"
    alpha_pitch_deg: float
    beta_yaw_deg: float
    gamma_roll_deg: float
    dx_mm: float
    dy_mm: float
    radius_d_mm: float
    target_cartesian_pose: List[float]     # [X, Y, Z, Rx, Ry, Rz, Fig] em mm e graus
    target_matrix_4x4: List[List[float]]   # ^B T_{T,k}^{alvo} (translação em mm)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_charuco_board(spec: CharucoBoardSpec) -> Tuple[cv2.aruco.Dictionary, cv2.aruco.CharucoBoard]:
    """Instancia o dicionário ArUco e o objeto cv2.aruco.CharucoBoard."""
    dict_id = ARUCO_DICT_MAP.get(spec.dictionary, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        float(spec.square_length_m),
        float(spec.marker_length_m),
        aruco_dict,
    )
    return aruco_dict, board


def generate_charuco_board_files(
    spec: CharucoBoardSpec,
    output_image_path: Union[str, Path] = "assets/charuco_board.png",
    output_spec_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    margin_mm: float = 10.0,
) -> Tuple[Path, Path]:
    """Gera a imagem PNG de alta resolução do tabuleiro ChArUco e o arquivo charuco_spec.json para impressão 1:1."""
    _, board = build_charuco_board(spec)
    img_path = Path(output_image_path)
    img_path.parent.mkdir(parents=True, exist_ok=True)

    spec_path = Path(output_spec_path) if output_spec_path else (img_path.parent / "charuco_spec.json")

    px_per_mm = dpi / 25.4
    width_px = int(round((spec.board_width_mm + 2.0 * margin_mm) * px_per_mm))
    height_px = int(round((spec.board_height_mm + 2.0 * margin_mm) * px_per_mm))
    margin_px = int(round(margin_mm * px_per_mm))

    board_img = board.generateImage((width_px, height_px), marginSize=margin_px, borderBits=1)
    cv2.imwrite(str(img_path), board_img)

    spec_payload = {
        **spec.to_dict(),
        "board_width_mm": spec.board_width_mm,
        "board_height_mm": spec.board_height_mm,
        "margin_mm": margin_mm,
        "total_print_width_mm": round(spec.board_width_mm + 2.0 * margin_mm, 2),
        "total_print_height_mm": round(spec.board_height_mm + 2.0 * margin_mm, 2),
        "dpi": dpi,
        "image_resolution_px": [width_px, height_px],
        "interior_corners_count": (spec.squares_x - 1) * (spec.squares_y - 1),
        "print_instructions": "Imprimir em escala 100% (1:1 real, sem ajuste à página) sobre superfície rígida e plana.",
    }
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec_payload, f, indent=2, ensure_ascii=False)

    return img_path, spec_path


class SphericalDomeGenerator:
    """Gerador das 24 poses da Abóbada Esférica Centrada no Alvo (Target-Centered Spherical Dome) em SE(3)."""

    def __init__(
        self,
        d0_mm: float = 350.0,
        inner_tilt_deg: float = 15.0,
        outer_tilt_deg: float = 22.0,
        outer_rolls_deg: Sequence[float] = (20.0, -20.0, 35.0, -35.0),
        outer_radii_factors: Sequence[float] = (0.85, 1.15),
        edge_tilt_deg: float = 8.0,
        edge_offset_fraction: float = 0.12,
        edge_radius_factor: float = 0.90,
        euler_convention: str = "xyz",
    ):
        self.d0_mm = float(d0_mm)
        self.inner_tilt_deg = float(inner_tilt_deg)
        self.outer_tilt_deg = float(outer_tilt_deg)
        self.outer_rolls_deg = [float(r) for r in outer_rolls_deg]
        self.outer_radii_factors = [float(f) for f in outer_radii_factors]
        self.edge_tilt_deg = float(edge_tilt_deg)
        self.edge_offset_fraction = float(edge_offset_fraction)
        self.edge_radius_factor = float(edge_radius_factor)
        self.euler_convention = euler_convention

    def compute_pivot_matrix(self, p_cal_0: Union[BasePose, Pose, Sequence[float]]) -> np.ndarray:
        """Calcula o pivô virtual no centro do tabuleiro:
            ^B T_{pivot} = ^B T_T(0) * [I, [0, 0, d_0]^T; 0, 1]
        """
        p0 = p_cal_0.to_pose() if isinstance(p_cal_0, BasePose) else ensure_pose(p_cal_0)
        t_b_t0 = pose_to_matrix(p0, euler_convention=self.euler_convention)
        t_shift = np.eye(4, dtype=np.float64)
        t_shift[2, 3] = self.d0_mm
        return t_b_t0 @ t_shift

    def compute_orbit_pose_matrix(
        self,
        t_b_pivot: np.ndarray,
        alpha_deg: float,
        beta_deg: float,
        gamma_deg: float,
        dx_mm: float,
        dy_mm: float,
        d_k_mm: float,
    ) -> np.ndarray:
        """Aplica a equação da órbita esférica apontando para o pivô:
            ^B T_{T,k}^{alvo} = ^B T_{pivot} * [R_orb(alpha, beta, gamma), [dx, dy, 0]^T; 0, 1] * [I, [0, 0, -d_k]^T; 0, 1]

        Quando alpha = beta = gamma = 0, dx = dy = 0 e d_k = d_0, retorna exatamente ^B T_T(0).
        """
        r_orb = Rotation.from_euler("xyz", [alpha_deg, beta_deg, gamma_deg], degrees=True).as_matrix()

        t_orb = np.eye(4, dtype=np.float64)
        t_orb[:3, :3] = r_orb
        t_orb[:3, 3] = [float(dx_mm), float(dy_mm), 0.0]

        t_back = np.eye(4, dtype=np.float64)
        t_back[2, 3] = -float(d_k_mm)

        return t_b_pivot @ t_orb @ t_back

    def generate_dome_poses(
        self,
        p_cal_0: Union[BasePose, Pose, Sequence[float]],
    ) -> Tuple[List[CalibrationDomePose], np.ndarray]:
        """Gera os 3 blocos funcionais totalizando 24 poses na abóbada esférica centrada no alvo."""
        p0 = p_cal_0.to_pose() if isinstance(p_cal_0, BasePose) else ensure_pose(p_cal_0)
        t_b_pivot = self.compute_pivot_matrix(p0)

        poses: List[CalibrationDomePose] = []
        pose_idx = 0

        # ------------------------------------------------------------------
        # Bloco 1 — Anel Interno / Desacoplamento Focal (8 poses)
        # d_k = 1.0 * d_0, gamma_k = 0, dx = dy = 0
        # 4 inclinações cardeais + 4 diagonais em (alpha, beta)
        # ------------------------------------------------------------------
        t1 = self.inner_tilt_deg
        block1_angles = [
            (+t1, 0.0),
            (-t1, 0.0),
            (0.0, +t1),
            (0.0, -t1),
            (+t1, +t1),
            (+t1, -t1),
            (-t1, +t1),
            (-t1, -t1),
        ]
        for alpha, beta in block1_angles:
            pose_idx += 1
            pose_id = f"cal_b1_{pose_idx:02d}"
            d_k = 1.0 * self.d0_mm
            t_mat = self.compute_orbit_pose_matrix(
                t_b_pivot, alpha, beta, 0.0, 0.0, 0.0, d_k
            )
            pose_obj = matrix_to_pose(t_mat, fig=p0.fig, euler_convention=self.euler_convention)
            poses.append(
                CalibrationDomePose(
                    pose_index=pose_idx,
                    pose_id=pose_id,
                    block_id=1,
                    block_name="inner_ring_focal",
                    filename=f"{pose_id}.jpg",
                    alpha_pitch_deg=alpha,
                    beta_yaw_deg=beta,
                    gamma_roll_deg=0.0,
                    dx_mm=0.0,
                    dy_mm=0.0,
                    radius_d_mm=round(d_k, 4),
                    target_cartesian_pose=[round(float(v), 6) for v in pose_obj.to_list()],
                    target_matrix_4x4=matrix_to_row_major_list(t_mat),
                )
            )

        # ------------------------------------------------------------------
        # Bloco 2 — Anel Externo / Excitação 6D para Hand-Eye AX = XB (8 poses)
        # Alterna d_k in {0.85 * d_0, 1.15 * d_0}, inclinações fortes + roll gamma_k
        # ------------------------------------------------------------------
        t2 = self.outer_tilt_deg
        r0, r1, r2, r3 = (
            self.outer_rolls_deg[0],
            self.outer_rolls_deg[1 % len(self.outer_rolls_deg)],
            self.outer_rolls_deg[2 % len(self.outer_rolls_deg)],
            self.outer_rolls_deg[3 % len(self.outer_rolls_deg)],
        )
        f_near = self.outer_radii_factors[0]
        f_far = self.outer_radii_factors[1 % len(self.outer_radii_factors)]

        block2_specs = [
            (+t2, 0.0, r0, f_near),
            (-t2, 0.0, r1, f_far),
            (0.0, +t2, r2, f_near),
            (0.0, -t2, r3, f_far),
            (+t2, +t2, r2, f_far),
            (+t2, -t2, r3, f_near),
            (-t2, +t2, r0, f_near),
            (-t2, -t2, r1, f_far),
        ]
        for alpha, beta, gamma, r_factor in block2_specs:
            pose_idx += 1
            pose_id = f"cal_b2_{pose_idx:02d}"
            d_k = r_factor * self.d0_mm
            t_mat = self.compute_orbit_pose_matrix(
                t_b_pivot, alpha, beta, gamma, 0.0, 0.0, d_k
            )
            pose_obj = matrix_to_pose(t_mat, fig=p0.fig, euler_convention=self.euler_convention)
            poses.append(
                CalibrationDomePose(
                    pose_index=pose_idx,
                    pose_id=pose_id,
                    block_id=2,
                    block_name="outer_ring_hand_eye",
                    filename=f"{pose_id}.jpg",
                    alpha_pitch_deg=alpha,
                    beta_yaw_deg=beta,
                    gamma_roll_deg=gamma,
                    dx_mm=0.0,
                    dy_mm=0.0,
                    radius_d_mm=round(d_k, 4),
                    target_cartesian_pose=[round(float(v), 6) for v in pose_obj.to_list()],
                    target_matrix_4x4=matrix_to_row_major_list(t_mat),
                )
            )

        # ------------------------------------------------------------------
        # Bloco 3 — Varredura de Bordas e Cantos / Distorção Radial (8 poses)
        # d_k = 0.90 * d_0, inclinações leves (+/- 8 deg) e deslocamentos laterais (dx, dy)
        # ------------------------------------------------------------------
        t3 = self.edge_tilt_deg
        shift_mm = round(self.edge_offset_fraction * self.d0_mm, 4)
        d_b3 = self.edge_radius_factor * self.d0_mm

        block3_specs = [
            (+t3, 0.0, +shift_mm, 0.0),          # Borda direita
            (-t3, 0.0, -shift_mm, 0.0),          # Borda esquerda
            (0.0, +t3, 0.0, +shift_mm),          # Borda inferior
            (0.0, -t3, 0.0, -shift_mm),          # Borda superior
            (+t3, +t3, +shift_mm, +shift_mm),    # Canto 1
            (+t3, -t3, +shift_mm, -shift_mm),    # Canto 2
            (-t3, +t3, -shift_mm, +shift_mm),    # Canto 3
            (-t3, -t3, -shift_mm, -shift_mm),    # Canto 4
        ]
        for alpha, beta, dx, dy in block3_specs:
            pose_idx += 1
            pose_id = f"cal_b3_{pose_idx:02d}"
            t_mat = self.compute_orbit_pose_matrix(
                t_b_pivot, alpha, beta, 0.0, dx, dy, d_b3
            )
            pose_obj = matrix_to_pose(t_mat, fig=p0.fig, euler_convention=self.euler_convention)
            poses.append(
                CalibrationDomePose(
                    pose_index=pose_idx,
                    pose_id=pose_id,
                    block_id=3,
                    block_name="edge_corner_distortion",
                    filename=f"{pose_id}.jpg",
                    alpha_pitch_deg=alpha,
                    beta_yaw_deg=beta,
                    gamma_roll_deg=0.0,
                    dx_mm=dx,
                    dy_mm=dy,
                    radius_d_mm=round(d_b3, 4),
                    target_cartesian_pose=[round(float(v), 6) for v in pose_obj.to_list()],
                    target_matrix_4x4=matrix_to_row_major_list(t_mat),
                )
            )

        return poses, t_b_pivot
