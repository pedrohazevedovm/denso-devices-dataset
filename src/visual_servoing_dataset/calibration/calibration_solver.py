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
