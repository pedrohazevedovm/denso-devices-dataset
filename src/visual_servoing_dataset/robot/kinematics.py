"""Módulo de cinemática em SE(3) para o robô DENSO VS-050.

Proíbe a soma direta de offsets nas coordenadas da base do robô.
Todas as perturbações P1..P7 são definidas no referencial local do TCP na pose inicial P0 (T_0)
e aplicadas por pós-multiplicação em SE(3):
    ^B T_{T,i}^{alvo} = ^B T_T(0) * Delta T_{T,i}
"""

from typing import List, Sequence, Tuple, Union
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import Pose, TCPOffset, ensure_pose


def pose_to_matrix(
    pose: Union[Pose, Sequence[float]],
    euler_convention: str = "xyz",
    degrees: bool = True,
) -> np.ndarray:
    """Converte um objeto Pose(X, Y, Z, Rx, Ry, Rz, Fig) do DENSO para matriz homogênea 4x4 em SE(3).

    A convenção padrão do controlador DENSO RC8 é Euler extrínseco 'xyz' (equivalente a intrínseco 'ZYX'),
    onde Rx é rotação em torno de X, Ry em torno de Y e Rz em torno de Z, em graus, e X, Y, Z em milímetros.
    """
    p = ensure_pose(pose) if not isinstance(pose, Pose) else pose
    rot = Rotation.from_euler(euler_convention, [p.rx, p.ry, p.rz], degrees=degrees)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = [float(p.x), float(p.y), float(p.z)]
    return mat


def matrix_to_pose(
    matrix: np.ndarray,
    fig: int = 0,
    euler_convention: str = "xyz",
    degrees: bool = True,
) -> Pose:
    """Converte uma matriz homogênea 4x4 em SE(3) de volta para o objeto Pose do DENSO preservando Fig."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Matriz homogênea deve ter dimensão (4, 4), recebido: {mat.shape}")

    rot = Rotation.from_matrix(mat[:3, :3])
    angles = rot.as_euler(euler_convention, degrees=degrees)
    x, y, z = mat[:3, 3]
    return Pose(
        x=float(x),
        y=float(y),
        z=float(z),
        rx=float(angles[0]),
        ry=float(angles[1]),
        rz=float(angles[2]),
        fig=int(fig),
    )


def tcp_offset_to_matrix(
    tcp: TCPOffset,
    euler_convention: str = "xyz",
    degrees: bool = True,
) -> np.ndarray:
    """Converte a definição de TCPOffset para a matriz homogênea 4x4 ^F T_T (Flange para TCP)."""
    rot = Rotation.from_euler(euler_convention, [tcp.rx, tcp.ry, tcp.rz], degrees=degrees)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = [float(tcp.x), float(tcp.y), float(tcp.z)]
    return mat


def invert_se3(matrix: np.ndarray) -> np.ndarray:
    """Calcula a inversa analítica exata em SE(3) de uma transformação rígida 4x4:
    T^{-1} = [R^T, -R^T * t; 0, 1].
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Matriz homogênea deve ter dimensão (4, 4), recebido: {mat.shape}")

    r_mat = mat[:3, :3]
    t_vec = mat[:3, 3]
    r_inv = r_mat.T
    t_inv = -r_inv @ t_vec

    inv_mat = np.eye(4, dtype=np.float64)
    inv_mat[:3, :3] = r_inv
    inv_mat[:3, 3] = t_inv
    return inv_mat


def perturbation_to_matrix(
    xi: Sequence[float],
    euler_convention: str = "xyz",
    degrees: bool = True,
) -> np.ndarray:
    """Constrói a matriz de perturbação local Delta T_{T,i} a partir de xi = [dx, dy, dz, drx, dry, drz]."""
    if len(xi) != 6:
        raise ValueError(f"Vetor de perturbação xi deve ter 6 elementos [dx, dy, dz, drx, dry, drz], recebido: {len(xi)}")

    dx, dy, dz, drx, dry, drz = [float(v) for v in xi]
    rot = Rotation.from_euler(euler_convention, [drx, dry, drz], degrees=degrees)
    delta_t = np.eye(4, dtype=np.float64)
    delta_t[:3, :3] = rot.as_matrix()
    delta_t[:3, 3] = [dx, dy, dz]
    return delta_t


def matrix_to_perturbation(
    delta_matrix: np.ndarray,
    euler_convention: str = "xyz",
    degrees: bool = True,
) -> List[float]:
    """Extrai o vetor de perturbação xi = [dx, dy, dz, drx, dry, drz] de uma matriz Delta T 4x4."""
    mat = np.asarray(delta_matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Matriz Delta T deve ter dimensão (4, 4), recebido: {mat.shape}")

    rot = Rotation.from_matrix(mat[:3, :3])
    angles = rot.as_euler(euler_convention, degrees=degrees)
    dx, dy, dz = mat[:3, 3]
    return [
        float(dx),
        float(dy),
        float(dz),
        float(angles[0]),
        float(angles[1]),
        float(angles[2]),
    ]


def apply_local_perturbation(
    base_pose_p0: Union[Pose, Sequence[float]],
    xi: Sequence[float],
    euler_convention: str = "xyz",
    degrees: bool = True,
) -> Tuple[Pose, np.ndarray, np.ndarray]:
    """Calcula a pose-alvo na base por pós-multiplicação em SE(3):
        ^B T_{T,i}^{alvo} = ^B T_T(0) * Delta T_{T,i}

    Retorna:
        (target_pose, target_matrix_4x4, delta_matrix_4x4)
    """
    p0 = ensure_pose(base_pose_p0) if not isinstance(base_pose_p0, Pose) else base_pose_p0
    t_b_t0 = pose_to_matrix(p0, euler_convention=euler_convention, degrees=degrees)
    delta_t = perturbation_to_matrix(xi, euler_convention=euler_convention, degrees=degrees)
    t_b_ti = t_b_t0 @ delta_t
    target_pose = matrix_to_pose(t_b_ti, fig=p0.fig, euler_convention=euler_convention, degrees=degrees)
    return target_pose, t_b_ti, delta_t


def compute_relative_transform(
    base_pose_p0: Union[Pose, Sequence[float], np.ndarray],
    measured_pose_pk: Union[Pose, Sequence[float], np.ndarray],
    euler_convention: str = "xyz",
    degrees: bool = True,
) -> Tuple[np.ndarray, List[float]]:
    """Calcula a transformação relativa real executada no referencial T_0:
        Delta T_{T,k}^{medido} = (^B T_T(0))^{-1} * ^B T_T(k)

    Retorna:
        (delta_matrix_4x4, measured_xi_6d)
    """
    if isinstance(base_pose_p0, np.ndarray) and base_pose_p0.shape == (4, 4):
        t_b_t0 = base_pose_p0
    else:
        t_b_t0 = pose_to_matrix(base_pose_p0, euler_convention=euler_convention, degrees=degrees)  # type: ignore[arg-type]

    if isinstance(measured_pose_pk, np.ndarray) and measured_pose_pk.shape == (4, 4):
        t_b_tk = measured_pose_pk
    else:
        t_b_tk = pose_to_matrix(measured_pose_pk, euler_convention=euler_convention, degrees=degrees)  # type: ignore[arg-type]

    delta_measured = invert_se3(t_b_t0) @ t_b_tk
    xi_measured = matrix_to_perturbation(delta_measured, euler_convention=euler_convention, degrees=degrees)
    return delta_measured, xi_measured


def compute_pose_error(
    pose_a: Union[Pose, np.ndarray],
    pose_b: Union[Pose, np.ndarray],
    euler_convention: str = "xyz",
) -> Tuple[float, float]:
    """Calcula a norma do desvio translacional (mm) e angular (graus) entre duas poses em SE(3)."""
    delta_mat, _ = compute_relative_transform(pose_a, pose_b, euler_convention=euler_convention)
    trans_err_mm = float(np.linalg.norm(delta_mat[:3, 3]))
    rot_err_deg = float(Rotation.from_matrix(delta_mat[:3, :3]).magnitude() * (180.0 / np.pi))
    return trans_err_mm, rot_err_deg


def matrix_to_row_major_list(matrix: np.ndarray) -> List[List[float]]:
    """Converte uma matriz 4x4 numpy para lista aninhada 4x4 row-major serializável em JSON."""
    mat = np.asarray(matrix, dtype=np.float64)
    return [[round(float(val), 8) for val in row] for row in mat]
