"""Fase 0: Solucionador '2 em 1' (Intrínsecos + Distorção + Hand-Eye AX=XB via ChArUco) e Retificação por LUT."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..robot.kinematics import invert_se3, matrix_to_row_major_list
from ..utils.logger import logger
from .calibration_generator import CharucoBoardSpec, build_charuco_board


HAND_EYE_METHODS: Dict[str, int] = {
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "CALIB_HAND_EYE_PARK": cv2.CALIB_HAND_EYE_PARK,
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "CALIB_HAND_EYE_TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    "CALIB_HAND_EYE_DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "CALIB_HAND_EYE_HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "CALIB_HAND_EYE_ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
}


@dataclass
class CharucoFrameDetection:
    """Resultado da detecção subpixel ChArUco para um único quadro de calibração."""
    pose_id: str
    valid: bool
    corners_count: int
    charuco_corners_2d: Optional[np.ndarray]  # (M, 2) float64
    charuco_ids: Optional[np.ndarray]         # (M,) int32
    object_points_3d_mm: Optional[np.ndarray] # (M, 3) float64 em milímetros
    t_base_to_tcp_mm: np.ndarray              # 4x4 ^B T_{T,k} (translação em mm)
    rejection_reason: Optional[str] = None


@dataclass
class CharucoCenteringMeasurement:
    """Medição geométrica para centralização (ID=8) e paralelismo por perímetros das extremidades (15,16 vs 0,1)."""
    valid: bool
    detected_markers_count: int
    center_marker_id: int
    center_marker_detected: bool
    center_u_px: float
    center_v_px: float
    image_center_u_px: float
    image_center_v_px: float
    error_u_px: float
    error_v_px: float
    center_error_norm_px: float
    center_marker_perimeter_px: float
    scale_mm_per_px: float
    error_u_mm: float
    error_v_mm: float
    center_error_norm_mm: float
    sum_left_perimeters_px: float
    sum_right_perimeters_px: float
    horiz_asymmetry_pct: float
    sum_top_perimeters_px: float
    sum_bottom_perimeters_px: float
    vert_asymmetry_pct: float
    vert_sensitivity_pct_per_deg: float
    roll_error_deg: float
    left_markers_used: List[int]
    right_markers_used: List[int]
    top_markers_used: List[int]
    bottom_markers_used: List[int]
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "detected_markers_count": self.detected_markers_count,
            "center_marker_id": self.center_marker_id,
            "center_marker_detected": self.center_marker_detected,
            "center_uv_px": [round(self.center_u_px, 3), round(self.center_v_px, 3)],
            "image_center_uv_px": [round(self.image_center_u_px, 3), round(self.image_center_v_px, 3)],
            "error_uv_px": [round(self.error_u_px, 3), round(self.error_v_px, 3)],
            "center_error_norm_px": round(self.center_error_norm_px, 3),
            "center_marker_perimeter_px": round(self.center_marker_perimeter_px, 3),
            "scale_mm_per_px": round(self.scale_mm_per_px, 6),
            "error_uv_mm": [round(self.error_u_mm, 3), round(self.error_v_mm, 3)],
            "center_error_norm_mm": round(self.center_error_norm_mm, 3),
            "sum_left_perimeters_px": round(self.sum_left_perimeters_px, 3),
            "sum_right_perimeters_px": round(self.sum_right_perimeters_px, 3),
            "horiz_asymmetry_pct": round(self.horiz_asymmetry_pct, 4),
            "sum_top_perimeters_px": round(self.sum_top_perimeters_px, 3),
            "sum_bottom_perimeters_px": round(self.sum_bottom_perimeters_px, 3),
            "vert_asymmetry_pct": round(self.vert_asymmetry_pct, 4),
            "roll_error_deg": round(self.roll_error_deg, 4),
            "left_markers_used": self.left_markers_used,
            "right_markers_used": self.right_markers_used,
            "top_markers_used": self.top_markers_used,
            "bottom_markers_used": self.bottom_markers_used,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class CalibrationSolution:
    """Resultado consolidado da calibração Intrínseca + Hand-Eye + Plano na Base."""
    resolution_px: Tuple[int, int]
    k_matrix: np.ndarray                      # 3x3
    k_rectified: np.ndarray                   # 3x3
    distortion_coeffs: np.ndarray             # (5,) ou (8,)
    distortion_model: str
    camera2_initial_guess_used: bool
    reprojection_rmse_px: float
    per_view_reprojection_rmse_px: List[float]
    hand_eye_method: str
    t_tcp_to_cam_mm: np.ndarray               # 4x4 ^T T_C (translação em mm)
    t_tcp_to_cam_m: np.ndarray                # 4x4 ^T T_C (translação em m)
    t_flange_to_cam_mm: np.ndarray            # 4x4 ^F T_C (translação em mm)
    t_flange_to_cam_m: np.ndarray             # 4x4 ^F T_C (translação em m)
    t_base_to_board_mean_mm: np.ndarray       # 4x4 ^B \bar{T}_M (translação em mm)
    t_base_to_board_mean_m: np.ndarray        # 4x4 ^B \bar{T}_M (translação em m)
    plane_normal_in_base: List[float]         # Vetor normal unitário do tabuleiro na base
    plane_distance_in_base_m: float           # Distância do plano à origem da base em metros
    valid_poses_count: int
    board_translation_rmse_mm: float
    board_translation_std_mm: List[float]
    board_rotation_rmse_deg: float
    alternative_methods_comparison: Dict[str, Dict[str, float]]


def scale_se3_translation(matrix_4x4: np.ndarray, factor: float) -> np.ndarray:
    """Escala o bloco de translação de uma matriz 4x4 em SE(3) (ex: mm -> m com factor=0.001)."""
    out = np.array(matrix_4x4, dtype=np.float64, copy=True)
    out[:3, 3] *= float(factor)
    return out


def _are_points_non_collinear(points_2d: np.ndarray, min_singular_ratio: float = 0.05) -> bool:
    """Verifica se os cantos detectados possuem dispersão bidimensional (não colineares)."""
    pts = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return False
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    _, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
    if s_vals[0] <= 1e-6:
        return False
    return bool((s_vals[1] / s_vals[0]) >= min_singular_ratio)


class CharucoCalibrationSolver:
    """Solucionador '2 em 1' para calibração intrínseca (K, D) e Hand-Eye (AX = XB) usando ChArUco."""

    def __init__(
        self,
        charuco_spec: CharucoBoardSpec,
        min_charuco_corners: int = 8,
        hand_eye_method: str = "PARK",
        num_distortion_coeffs: int = 5,
    ):
        self.charuco_spec = charuco_spec
        self.min_charuco_corners = int(min_charuco_corners)
        self.hand_eye_method = hand_eye_method.upper()
        self.num_distortion_coeffs = int(num_distortion_coeffs)

        self.aruco_dict, self.board = build_charuco_board(charuco_spec)
        self._detector_params = cv2.aruco.DetectorParameters()
        self._detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(self.board, detectorParams=self._detector_params)

        # Coordenadas 3D dos cantos internos do tabuleiro ChArUco em milímetros (Z = 0)
        self.board_corners_3d_mm: np.ndarray = (
            np.asarray(self.board.getChessboardCorners(), dtype=np.float64) * 1000.0
        )

    def detect_charuco_in_image(
        self,
        image_bgr: np.ndarray,
        pose_id: str,
        t_base_to_tcp_mm: np.ndarray,
        debug_save_path: Optional[Path] = None,
    ) -> CharucoFrameDetection:
        """Etapa A: Detecta cantos ChArUco com refinamento subpixel e valida contagem mínima e não-colinearidade."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

        # Fallback inteligente: se o tabuleiro impresso tiver orientação transposta (ex: 5x7 em vez de 7x5)
        count_primary = 0 if charuco_ids is None else int(len(charuco_ids))
        if count_primary < self.min_charuco_corners and self.charuco_spec.squares_x != self.charuco_spec.squares_y:
            transposed_spec = CharucoBoardSpec(
                squares_x=self.charuco_spec.squares_y,
                squares_y=self.charuco_spec.squares_x,
                square_length_m=self.charuco_spec.square_length_m,
                marker_length_m=self.charuco_spec.marker_length_m,
                dictionary=self.charuco_spec.dictionary,
            )
            _, t_board = build_charuco_board(transposed_spec)
            t_detector = cv2.aruco.CharucoDetector(t_board, detectorParams=self._detector_params)
            t_corners, t_ids, t_mcorners, t_mids = t_detector.detectBoard(gray)
            count_transposed = 0 if t_ids is None else int(len(t_ids))
            if count_transposed >= self.min_charuco_corners and count_transposed > count_primary:
                logger.info(
                    f"Auto-detectada orientação transposta do tabuleiro ChArUco impresso: "
                    f"({transposed_spec.squares_x}x{transposed_spec.squares_y}) em vez de "
                    f"({self.charuco_spec.squares_x}x{self.charuco_spec.squares_y}) "
                    f"({count_transposed} cantos detectados em {pose_id})."
                )
                self.charuco_spec = transposed_spec
                self.board = t_board
                self.detector = t_detector
                self.board_corners_3d_mm = (
                    np.asarray(self.board.getChessboardCorners(), dtype=np.float64) * 1000.0
                )
                charuco_corners, charuco_ids, marker_corners, marker_ids = (
                    t_corners,
                    t_ids,
                    t_mcorners,
                    t_mids,
                )

        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < self.min_charuco_corners:
            count = 0 if charuco_ids is None else int(len(charuco_ids))
            if debug_save_path is not None:
                debug_img = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
                if marker_corners is not None and len(marker_corners) > 0:
                    cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
                if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids, (0, 165, 255))
                debug_save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_save_path), debug_img)
            return CharucoFrameDetection(
                pose_id=pose_id,
                valid=False,
                corners_count=count,
                charuco_corners_2d=None,
                charuco_ids=None,
                object_points_3d_mm=None,
                t_base_to_tcp_mm=np.asarray(t_base_to_tcp_mm, dtype=np.float64),
                rejection_reason=f"Cantos insuficientes ({count} < {self.min_charuco_corners})",
            )

        corners_2d = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        ids_1d = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        obj_pts_mm = self.board_corners_3d_mm[ids_1d]

        if not _are_points_non_collinear(corners_2d):
            return CharucoFrameDetection(
                pose_id=pose_id,
                valid=False,
                corners_count=int(len(ids_1d)),
                charuco_corners_2d=None,
                charuco_ids=None,
                object_points_3d_mm=None,
                t_base_to_tcp_mm=np.asarray(t_base_to_tcp_mm, dtype=np.float64),
                rejection_reason="Cantos detectados quase colineares",
            )

        if debug_save_path is not None:
            debug_img = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            if marker_corners is not None and len(marker_corners) > 0:
                cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids, (0, 255, 0))
            debug_save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_save_path), debug_img)

        return CharucoFrameDetection(
            pose_id=pose_id,
            valid=True,
            corners_count=int(len(ids_1d)),
            charuco_corners_2d=corners_2d,
            charuco_ids=ids_1d,
            object_points_3d_mm=obj_pts_mm,
            t_base_to_tcp_mm=np.asarray(t_base_to_tcp_mm, dtype=np.float64),
        )

    def analyze_charuco_centering_and_parallelism(
        self,
        image_bgr: np.ndarray,
        center_marker_id: int = 8,
        debug_save_path: Optional[Path] = None,
    ) -> CharucoCenteringMeasurement:
        """
        Analisa a posição do marcador central (ID=8) e os perímetros dos marcadores das extremidades
        (Esquerda: 15,16 vs Direita: 0,1 | Topo: 2,7,12/15,0 vs Base: 4,9,14/16,1) para centralização
        e alinhamento de paralelismo em malha fechada.
        """
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        img_cu, img_cv = width / 2.0, height / 2.0
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr

        aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self._detector_params)
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) < 3:
            cnt = 0 if marker_ids is None else int(len(marker_ids))
            return CharucoCenteringMeasurement(
                valid=False,
                detected_markers_count=cnt,
                center_marker_id=center_marker_id,
                center_marker_detected=False,
                center_u_px=img_cu,
                center_v_px=img_cv,
                image_center_u_px=img_cu,
                image_center_v_px=img_cv,
                error_u_px=0.0,
                error_v_px=0.0,
                center_error_norm_px=0.0,
                center_marker_perimeter_px=0.0,
                scale_mm_per_px=0.0,
                error_u_mm=0.0,
                error_v_mm=0.0,
                center_error_norm_mm=0.0,
                sum_left_perimeters_px=0.0,
                sum_right_perimeters_px=0.0,
                horiz_asymmetry_pct=0.0,
                sum_top_perimeters_px=0.0,
                sum_bottom_perimeters_px=0.0,
                vert_asymmetry_pct=0.0,
                vert_sensitivity_pct_per_deg=1.1,
                roll_error_deg=0.0,
                left_markers_used=[],
                right_markers_used=[],
                top_markers_used=[],
                bottom_markers_used=[],
                rejection_reason=f"Marcadores ArUco insuficientes detectados ({cnt} < 3)",
            )

        perimeters: Dict[int, float] = {}
        centers: Dict[int, np.ndarray] = {}
        corners_map: Dict[int, np.ndarray] = {}
        for c, mid_arr in zip(marker_corners, marker_ids.flatten()):
            mid = int(mid_arr)
            pts = np.asarray(c[0], dtype=np.float64)
            corners_map[mid] = pts
            centers[mid] = pts.mean(axis=0)
            perimeters[mid] = float(cv2.arcLength(pts.astype(np.float32), True))

        # 1. Centro do Tabuleiro via ArUco ID=8 (com fallback por homografia se ID=8 estiver ocluído)
        center_detected = center_marker_id in centers
        if center_detected:
            cu, cv = float(centers[center_marker_id][0]), float(centers[center_marker_id][1])
            p_center = float(perimeters[center_marker_id])
        else:
            obj_pts_all = self.board.getObjPoints()
            board_ids_all = np.asarray(self.board.getIds()).flatten()
            board_centers_3d = {
                int(bid): np.asarray(pts, dtype=np.float64).mean(axis=0)[:2] * 1000.0
                for bid, pts in zip(board_ids_all, obj_pts_all)
            }
            common_ids = [mid for mid in centers.keys() if mid in board_centers_3d]
            if len(common_ids) >= 4:
                src_pts = np.array([board_centers_3d[m] for m in common_ids], dtype=np.float64)
                dst_pts = np.array([centers[m] for m in common_ids], dtype=np.float64)
                h_mat, _ = cv2.findHomography(src_pts, dst_pts, method=0)
                target_3d = board_centers_3d.get(
                    center_marker_id,
                    np.array([self.charuco_spec.board_width_mm / 2.0, self.charuco_spec.board_height_mm / 2.0]),
                )
                if h_mat is not None:
                    proj = h_mat @ np.array([target_3d[0], target_3d[1], 1.0], dtype=np.float64)
                    cu, cv = float(proj[0] / proj[2]), float(proj[1] / proj[2])
                else:
                    mean_uv = np.mean(list(centers.values()), axis=0)
                    cu, cv = float(mean_uv[0]), float(mean_uv[1])
            else:
                mean_uv = np.mean(list(centers.values()), axis=0)
                cu, cv = float(mean_uv[0]), float(mean_uv[1])
            p_center = float(np.mean(list(perimeters.values())))

        marker_perimeter_mm = 4.0 * (self.charuco_spec.marker_length_m * 1000.0)
        scale_mm_per_px = marker_perimeter_mm / max(p_center, 1e-6)

        err_u_px = cu - img_cu
        err_v_px = cv - img_cv
        err_norm_px = float(np.hypot(err_u_px, err_v_px))
        err_u_mm = err_u_px * scale_mm_per_px
        err_v_mm = err_v_px * scale_mm_per_px
        err_norm_mm = float(np.hypot(err_u_mm, err_v_mm))

        # 2. Paralelismo Horizontal: Marcadores da extremidade esquerda (15, 16) vs direita (0, 1)
        left_used = [m for m in (15, 16) if m in perimeters]
        if not left_used:
            left_used = [m for m in (12, 13, 14) if m in perimeters]
        right_used = [m for m in (0, 1) if m in perimeters]
        if not right_used:
            right_used = [m for m in (2, 3, 4) if m in perimeters]

        if left_used and right_used:
            # Soma equivalente a 2 marcadores de cada lado
            sum_left = 2.0 * float(np.mean([perimeters[m] for m in left_used]))
            sum_right = 2.0 * float(np.mean([perimeters[m] for m in right_used]))
            horiz_asym_pct = 100.0 * (sum_left - sum_right) / (0.5 * (sum_left + sum_right))
        else:
            sum_left, sum_right, horiz_asym_pct = 0.0, 0.0, 0.0

        # 3. Paralelismo Vertical: Marcadores do topo (2, 7, 12 ou 15, 0) vs base (4, 9, 14 ou 16, 1)
        top_outer = [m for m in (2, 7, 12) if m in perimeters]
        bot_outer = [m for m in (4, 9, 14) if m in perimeters]
        if len(top_outer) >= 1 and len(bot_outer) >= 1:
            top_used = top_outer
            bot_used = bot_outer
            vert_sens = 1.15
        else:
            top_used = [m for m in (15, 10, 5, 0) if m in perimeters]
            bot_used = [m for m in (16, 11, 6, 1) if m in perimeters]
            vert_sens = 0.60

        if top_used and bot_used:
            sum_top = 2.0 * float(np.mean([perimeters[m] for m in top_used]))
            sum_bot = 2.0 * float(np.mean([perimeters[m] for m in bot_used]))
            vert_asym_pct = 100.0 * (sum_top - sum_bot) / (0.5 * (sum_top + sum_bot))
        else:
            sum_top, sum_bot, vert_asym_pct = 0.0, 0.0, 0.0

        # 4. Alinhamento de Roll no plano da imagem (linha central 13 -> 8 -> 3 ou borda do ID=8)
        roll_angles_deg: List[float] = []
        for m_left, m_right in ((13, 3), (13, 8), (8, 3), (15, 0), (16, 1), (12, 2), (14, 4)):
            if m_left in centers and m_right in centers:
                vec = centers[m_right] - centers[m_left]
                if np.linalg.norm(vec) > 10.0:
                    roll_angles_deg.append(float(np.degrees(np.arctan2(vec[1], vec[0]))))
        if not roll_angles_deg and center_marker_id in corners_map:
            c8 = corners_map[center_marker_id]
            # Em paisagem (5x7), a aresta c[1]->c[2] ou c[0]->c[3] aponta ao longo do eixo horizontal
            vec_edge = c8[0] - c8[3]
            roll_angles_deg.append(float(np.degrees(np.arctan2(vec_edge[1], vec_edge[0]))))
        roll_error_deg = float(np.median(roll_angles_deg)) if roll_angles_deg else 0.0

        if debug_save_path is not None:
            dbg = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(dbg, marker_corners, marker_ids)
            # Desenhar mira no centro da imagem (ciano) e no centro do ID=8 (verde)
            ic_pt = (int(round(img_cu)), int(round(img_cv)))
            c8_pt = (int(round(cu)), int(round(cv)))
            cv2.drawMarker(dbg, ic_pt, (255, 255, 0), cv2.MARKER_CROSS, 36, 2)
            cv2.circle(dbg, c8_pt, 10, (0, 255, 0), -1)
            cv2.arrowedLine(dbg, c8_pt, ic_pt, (0, 165, 255), 3, tipLength=0.15)
            info_lines = [
                f"ID={center_marker_id} err: ({err_u_px:+.1f}, {err_v_px:+.1f}) px | {err_norm_mm:.2f} mm",
                f"Perim Esq(15,16)={sum_left:.1f} vs Dir(0,1)={sum_right:.1f} px ({horiz_asym_pct:+.2f}%)",
                f"Perim Topo={sum_top:.1f} vs Base={sum_bot:.1f} px ({vert_asym_pct:+.2f}%) | Roll={roll_error_deg:+.2f} deg",
            ]
            for idx_line, text in enumerate(info_lines):
                cv2.putText(
                    dbg,
                    text,
                    (25, 40 + idx_line * 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            debug_save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_save_path), dbg)

        return CharucoCenteringMeasurement(
            valid=True,
            detected_markers_count=len(perimeters),
            center_marker_id=center_marker_id,
            center_marker_detected=center_detected,
            center_u_px=cu,
            center_v_px=cv,
            image_center_u_px=img_cu,
            image_center_v_px=img_cv,
            error_u_px=err_u_px,
            error_v_px=err_v_px,
            center_error_norm_px=err_norm_px,
            center_marker_perimeter_px=p_center,
            scale_mm_per_px=scale_mm_per_px,
            error_u_mm=err_u_mm,
            error_v_mm=err_v_mm,
            center_error_norm_mm=err_norm_mm,
            sum_left_perimeters_px=sum_left,
            sum_right_perimeters_px=sum_right,
            horiz_asymmetry_pct=horiz_asym_pct,
            sum_top_perimeters_px=sum_top,
            sum_bottom_perimeters_px=sum_bot,
            vert_asymmetry_pct=vert_asym_pct,
            vert_sensitivity_pct_per_deg=vert_sens,
            roll_error_deg=roll_error_deg,
            left_markers_used=left_used,
            right_markers_used=right_used,
            top_markers_used=top_used,
            bottom_markers_used=bot_used,
        )

    def solve_from_detections(
        self,
        detections: Sequence[CharucoFrameDetection],
        resolution_px: Tuple[int, int],
        t_flange_to_tcp_mm: Optional[np.ndarray] = None,
        camera2_metadata: Optional[Dict[str, Any]] = None,
    ) -> CalibrationSolution:
        """Executa as Etapas B, C e D sobre as vistas válidas para estimar K, D, ^T T_C, ^F T_C e ^B \bar{T}_M."""
        valid_dets = [
            d for d in detections
            if d.valid and d.charuco_corners_2d is not None and d.object_points_3d_mm is not None
        ]
        if len(valid_dets) < 6:
            raise ValueError(
                f"Calibração requer pelo menos 6 vistas válidas com cantos ChArUco; recebidas: {len(valid_dets)}"
            )

        width, height = int(resolution_px[0]), int(resolution_px[1])
        object_points_list: List[np.ndarray] = []
        image_points_list: List[np.ndarray] = []

        for d in valid_dets:
            assert d.object_points_3d_mm is not None
            assert d.charuco_corners_2d is not None
            object_points_list.append(d.object_points_3d_mm.astype(np.float32).reshape(-1, 1, 3))
            image_points_list.append(d.charuco_corners_2d.astype(np.float32).reshape(-1, 1, 2))

        # ------------------------------------------------------------------
        # Etapa B: Calibração Intrínseca e de Distorção (cv2.calibrateCamera)
        # ------------------------------------------------------------------
        initial_k, guess_used = self._extract_camera2_initial_guess(camera2_metadata, (width, height))
        calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS if guess_used else 0
        dist_init = np.zeros((self.num_distortion_coeffs, 1), dtype=np.float64)

        reproj_rmse, k_est, dist_est, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=object_points_list,
            imagePoints=image_points_list,
            imageSize=(width, height),
            cameraMatrix=initial_k if guess_used else None,  # type: ignore[arg-type]
            distCoeffs=dist_init,
            flags=calib_flags,
        )

        k_matrix = np.asarray(k_est, dtype=np.float64).reshape(3, 3)
        dist_flat = np.asarray(dist_est, dtype=np.float64).reshape(-1)[: self.num_distortion_coeffs]

        per_view_rmse: List[float] = []
        for idx, (obj_pts, img_pts) in enumerate(zip(object_points_list, image_points_list)):
            proj_pts, _ = cv2.projectPoints(obj_pts, rvecs[idx], tvecs[idx], k_matrix, dist_flat)
            err = np.linalg.norm(proj_pts.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)
            view_rmse = float(np.sqrt(np.mean(err ** 2)))
            per_view_rmse.append(round(view_rmse, 6))

        # ------------------------------------------------------------------
        # Etapa C: Estimação de Pose por Vista (solvePnP) e Hand-Eye (AX = XB)
        # ------------------------------------------------------------------
        r_gripper2base_list: List[np.ndarray] = []
        t_gripper2base_mm_list: List[np.ndarray] = []
        r_target2cam_list: List[np.ndarray] = []
        t_target2cam_mm_list: List[np.ndarray] = []
        t_cam_to_board_mm_list: List[np.ndarray] = []

        for idx, d in enumerate(valid_dets):
            assert d.object_points_3d_mm is not None
            assert d.charuco_corners_2d is not None
            obj_pts_f64 = d.object_points_3d_mm.astype(np.float64).reshape(-1, 1, 3)
            img_pts_f64 = d.charuco_corners_2d.astype(np.float64).reshape(-1, 1, 2)

            rvec_init = np.asarray(rvecs[idx], dtype=np.float64).reshape(3, 1)
            tvec_init = np.asarray(tvecs[idx], dtype=np.float64).reshape(3, 1)

            ok, rvec_ref, tvec_ref = cv2.solvePnP(
                obj_pts_f64,
                img_pts_f64,
                k_matrix,
                dist_flat,
                rvec=rvec_init,
                tvec=tvec_init,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                rvec_ref, tvec_ref = rvec_init, tvec_init

            r_cm, _ = cv2.Rodrigues(rvec_ref)
            t_cm_mm = np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1)

            t_c_m = np.eye(4, dtype=np.float64)
            t_c_m[:3, :3] = r_cm
            t_c_m[:3, 3] = t_cm_mm.reshape(3)
            t_cam_to_board_mm_list.append(t_c_m)

            r_target2cam_list.append(r_cm)
            t_target2cam_mm_list.append(t_cm_mm)

            t_b_t = d.t_base_to_tcp_mm
            r_gripper2base_list.append(t_b_t[:3, :3].copy())
            t_gripper2base_mm_list.append(t_b_t[:3, 3].reshape(3, 1).copy())

        primary_method_flag = HAND_EYE_METHODS.get(self.hand_eye_method, cv2.CALIB_HAND_EYE_PARK)
        primary_method_name = (
            self.hand_eye_method
            if self.hand_eye_method.startswith("CALIB_HAND_EYE_")
            else f"CALIB_HAND_EYE_{self.hand_eye_method}"
        )

        r_tc, t_tc_mm = cv2.calibrateHandEye(
            R_gripper2base=r_gripper2base_list,
            t_gripper2base=t_gripper2base_mm_list,
            R_target2cam=r_target2cam_list,
            t_target2cam=t_target2cam_mm_list,
            method=primary_method_flag,
        )

        t_tcp_to_cam_mm = np.eye(4, dtype=np.float64)
        t_tcp_to_cam_mm[:3, :3] = np.asarray(r_tc, dtype=np.float64)
        t_tcp_to_cam_mm[:3, 3] = np.asarray(t_tc_mm, dtype=np.float64).reshape(3)

        if t_flange_to_tcp_mm is not None:
            t_f_t = np.asarray(t_flange_to_tcp_mm, dtype=np.float64)
        else:
            t_f_t = np.eye(4, dtype=np.float64)

        t_flange_to_cam_mm = t_f_t @ t_tcp_to_cam_mm

        # ------------------------------------------------------------------
        # Etapa D: Validação Cruzada, Pose Média do Tabuleiro na Base e Métricas
        # ^B T_{M,k} = ^B T_{T,k} * ^T T_C * ^C T_{M,k}
        # ------------------------------------------------------------------
        t_base_to_board_mean_mm, trans_rmse_mm, trans_std_mm, rot_rmse_deg = (
            self._evaluate_board_consistency(valid_dets, t_tcp_to_cam_mm, t_cam_to_board_mm_list)
        )

        # Comparação cruzada com outros métodos Hand-Eye (TSAI, PARK, DANIILIDIS)
        alt_comparison: Dict[str, Dict[str, float]] = {}
        for m_label, m_flag in (
            ("CALIB_HAND_EYE_PARK", cv2.CALIB_HAND_EYE_PARK),
            ("CALIB_HAND_EYE_TSAI", cv2.CALIB_HAND_EYE_TSAI),
            ("CALIB_HAND_EYE_DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
        ):
            try:
                r_alt, t_alt = cv2.calibrateHandEye(
                    R_gripper2base=r_gripper2base_list,
                    t_gripper2base=t_gripper2base_mm_list,
                    R_target2cam=r_target2cam_list,
                    t_target2cam=t_target2cam_mm_list,
                    method=m_flag,
                )
                t_alt_mat = np.eye(4, dtype=np.float64)
                t_alt_mat[:3, :3] = r_alt
                t_alt_mat[:3, 3] = np.asarray(t_alt, dtype=np.float64).reshape(3)
                _, alt_t_rmse, _, alt_r_rmse = self._evaluate_board_consistency(
                    valid_dets, t_alt_mat, t_cam_to_board_mm_list
                )
                alt_comparison[m_label] = {
                    "board_translation_rmse_mm": round(alt_t_rmse, 4),
                    "board_rotation_rmse_deg": round(alt_r_rmse, 4),
                }
            except Exception as exc:
                logger.debug(f"Método alternativo {m_label} falhou: {exc}")

        # Equação do plano do tabuleiro no referencial da base
        normal_base = t_base_to_board_mean_mm[:3, 2].copy()
        normal_norm = float(np.linalg.norm(normal_base))
        if normal_norm > 1e-12:
            normal_base /= normal_norm
        origin_board_m = t_base_to_board_mean_mm[:3, 3] / 1000.0
        plane_dist_m = float(abs(np.dot(normal_base, origin_board_m)))

        return CalibrationSolution(
            resolution_px=(width, height),
            k_matrix=k_matrix,
            k_rectified=k_matrix.copy(),
            distortion_coeffs=dist_flat,
            distortion_model=f"opencv_plumb_bob_{self.num_distortion_coeffs}",
            camera2_initial_guess_used=guess_used,
            reprojection_rmse_px=round(float(reproj_rmse), 6),
            per_view_reprojection_rmse_px=per_view_rmse,
            hand_eye_method=primary_method_name,
            t_tcp_to_cam_mm=t_tcp_to_cam_mm,
            t_tcp_to_cam_m=scale_se3_translation(t_tcp_to_cam_mm, 0.001),
            t_flange_to_cam_mm=t_flange_to_cam_mm,
            t_flange_to_cam_m=scale_se3_translation(t_flange_to_cam_mm, 0.001),
            t_base_to_board_mean_mm=t_base_to_board_mean_mm,
            t_base_to_board_mean_m=scale_se3_translation(t_base_to_board_mean_mm, 0.001),
            plane_normal_in_base=[round(float(v), 8) for v in normal_base],
            plane_distance_in_base_m=round(plane_dist_m, 6),
            valid_poses_count=len(valid_dets),
            board_translation_rmse_mm=round(trans_rmse_mm, 6),
            board_translation_std_mm=[round(float(v), 6) for v in trans_std_mm],
            board_rotation_rmse_deg=round(rot_rmse_deg, 6),
            alternative_methods_comparison=alt_comparison,
        )

    @staticmethod
    def _evaluate_board_consistency(
        valid_dets: Sequence[CharucoFrameDetection],
        t_tcp_to_cam_mm: np.ndarray,
        t_cam_to_board_mm_list: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """Calcula ^B T_{M,k} = ^B T_{T,k} * ^T T_C * ^C T_{M,k}, sua média em SE(3) e os erros RMSE/STD."""
        board_translations_mm: List[np.ndarray] = []
        board_rotations: List[np.ndarray] = []

        for d, t_c_m in zip(valid_dets, t_cam_to_board_mm_list):
            t_b_m = d.t_base_to_tcp_mm @ t_tcp_to_cam_mm @ t_c_m
            board_translations_mm.append(t_b_m[:3, 3])
            board_rotations.append(t_b_m[:3, :3])

        trans_arr = np.asarray(board_translations_mm, dtype=np.float64)
        mean_trans_mm = np.mean(trans_arr, axis=0)
        trans_std_mm = np.std(trans_arr, axis=0)
        trans_diffs = np.linalg.norm(trans_arr - mean_trans_mm, axis=1)
        trans_rmse_mm = float(np.sqrt(np.mean(trans_diffs ** 2)))

        rots_obj = Rotation.from_matrix(np.asarray(board_rotations, dtype=np.float64))
        mean_rot = rots_obj.mean()
        mean_rot_mat = mean_rot.as_matrix()

        ang_errors_deg: List[float] = []
        for r_k in board_rotations:
            r_diff = mean_rot_mat.T @ r_k
            ang_deg = float(Rotation.from_matrix(r_diff).magnitude() * (180.0 / np.pi))
            ang_errors_deg.append(ang_deg)

        rot_rmse_deg = float(np.sqrt(np.mean(np.asarray(ang_errors_deg) ** 2)))

        t_mean = np.eye(4, dtype=np.float64)
        t_mean[:3, :3] = mean_rot_mat
        t_mean[:3, 3] = mean_trans_mm
        return t_mean, trans_rmse_mm, trans_std_mm, rot_rmse_deg

    @staticmethod
    def _extract_camera2_initial_guess(
        metadata: Optional[Dict[str, Any]],
        resolution_px: Tuple[int, int],
    ) -> Tuple[np.ndarray, bool]:
        """Extrai chute inicial de K a partir de X-Camera-Metadata apenas se válido e coerente."""
        width, height = resolution_px
        default_k = np.array(
            [[float(width), 0.0, width / 2.0], [0.0, float(width), height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        if not metadata:
            return default_k, False

        k_flat = metadata.get("camera_matrix_k")
        if isinstance(k_flat, list) and len(k_flat) == 9:
            k_cand = np.asarray(k_flat, dtype=np.float64).reshape(3, 3)
            fx, fy, cx, cy = k_cand[0, 0], k_cand[1, 1], k_cand[0, 2], k_cand[1, 2]
            if fx > 100.0 and fy > 100.0 and 0.0 < cx < width and 0.0 < cy < height:
                return k_cand, True

        intr = metadata.get("lens_intrinsic_calibration")
        if isinstance(intr, list) and len(intr) >= 4:
            fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
            if fx > 100.0 and fy > 100.0 and 0.0 < cx < width and 0.0 < cy < height:
                k_cand = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
                return k_cand, True

        return default_k, False


class ImageRectifier:
    """Retificador de imagens e coordenadas 2D usando LUT pré-computada (cv2.initUndistortRectifyMap + cv2.remap)."""

    def __init__(
        self,
        k_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        distortion_coeffs: Union[np.ndarray, Sequence[float]],
        resolution_px: Tuple[int, int],
        k_rectified: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    ):
        self.k_matrix = np.asarray(k_matrix, dtype=np.float64).reshape(3, 3)
        self.distortion_coeffs = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1)
        self.resolution_px = (int(resolution_px[0]), int(resolution_px[1]))
        self.k_rectified = (
            np.asarray(k_rectified, dtype=np.float64).reshape(3, 3)
            if k_rectified is not None
            else self.k_matrix.copy()
        )
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.k_matrix,
            self.distortion_coeffs,
            None,
            self.k_rectified,
            self.resolution_px,
            cv2.CV_32FC1,
        )

    def rectify_image(self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Retifica uma imagem distorcida usando a LUT pré-computada via cv2.remap."""
        return cv2.remap(image, self.map_x, self.map_y, interpolation=interpolation)

    def rectify_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Remove a distorção de coordenadas 2D (N, 2) projetando-as no domínio pinhole puro K_rectified."""
        pts = np.asarray(points_2d, dtype=np.float64).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(pts, self.k_matrix, self.distortion_coeffs, R=None, P=self.k_rectified)
        return undist.reshape(-1, 2)


def compute_planar_homography_gt(
    K_rect: Union[np.ndarray, Sequence[Sequence[float]]],
    T_c1_c2: np.ndarray,
    plane_normal_c1: Sequence[float],
    plane_distance_c1: float,
) -> np.ndarray:
    """Calcula a homografia teórica pinhole no domínio retificado:
        H_gt ~ K_rect * (R_21 + (t_21 * n_1^T) / d_1) * K_rect^{-1}

    Convenção:
      - T_c1_c2 (4x4): Transformação que leva coordenadas do frame da câmera 1 (C1) para a câmera 2 (C2):
            X_{C2} = R_21 * X_{C1} + t_21
        (ou seja, T_21 = (^B T_{C2})^{-1} * ^B T_{C1}).
      - plane_normal_c1 (3,): Vetor normal unitário n_1 do plano expresso no frame C1.
      - plane_distance_c1 (float): Distância escalar d_1 tal que, para todo ponto X_{C1} no plano,
            n_1^T * X_{C1} = d_1  (ou seja, (n_1^T * X_{C1}) / d_1 = 1).
    """
    k_mat = np.asarray(K_rect, dtype=np.float64).reshape(3, 3)
    k_inv = np.linalg.inv(k_mat)

    t_mat = np.asarray(T_c1_c2, dtype=np.float64).reshape(4, 4)
    r_21 = t_mat[:3, :3]
    t_21 = t_mat[:3, 3].reshape(3, 1)

    n_1 = np.asarray(plane_normal_c1, dtype=np.float64).reshape(3, 1)
    d_1 = float(plane_distance_c1)
    if abs(d_1) < 1e-12:
        raise ValueError("plane_distance_c1 não pode ser zero.")

    euclidean_h = r_21 + (t_21 @ n_1.T) / d_1
    h_gt = k_mat @ euclidean_h @ k_inv
    if abs(h_gt[2, 2]) > 1e-12:
        h_gt = h_gt / h_gt[2, 2]
    return h_gt


def compute_plane_normal_and_distance_in_cam(
    t_base_to_cam: np.ndarray,
    t_base_to_board: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Calcula (plane_normal_c1, plane_distance_c1) de um plano Z_board=0 no referencial da câmera C1
    tal que n_1^T * X_{C1} = d_1 > 0 para qualquer ponto X_{C1} sobre o tabuleiro.
    """
    t_cam_to_board = invert_se3(t_base_to_cam) @ t_base_to_board
    n_c1 = t_cam_to_board[:3, 2].copy()
    norm_val = float(np.linalg.norm(n_c1))
    if norm_val > 1e-12:
        n_c1 /= norm_val

    t_c1_origin = t_cam_to_board[:3, 3]
    d_c1 = float(np.dot(n_c1, t_c1_origin))
    if d_c1 < 0.0:
        n_c1 = -n_c1
        d_c1 = -d_c1
    return n_c1, d_c1
