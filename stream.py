"""
Visualizador em Tempo Real (Live Stream) do Cam-App Android (estilo scrcpy).

Conecta ao servidor HTTP do Cam-App utilizando o IP definido no `config.yaml` (ou via `--ip`),
configura a câmera exclusivamente no modo `auto` e exibe o fluxo de vídeo com mira de
enquadramento, telemetria em tempo real (FPS, Foco, ISO, Exposição) e inspetor opcional de ChArUco.

Atalhos na janela de vídeo:
  [q] ou [ESC] : Sair
  [g]          : Alternar mira central e grade de enquadramento (ON/OFF)
  [c]          : Alternar detecção ao vivo de ChArUco / ArUco ID=8 (ON/OFF)
  [h]          : Alternar HUD de telemetria (ON/OFF)
  [r]          : Reaplicar modo 'auto' (refocar)
  [s]          : Salvar snapshot em assets/snapshots/
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from requests.exceptions import RequestException

from src.visual_servoing_dataset.calibration.calibration_generator import CharucoBoardSpec
from src.visual_servoing_dataset.calibration.calibration_solver import CharucoCalibrationSolver
from src.visual_servoing_dataset.config import AppConfig, load_config
from src.visual_servoing_dataset.smartphone.phone_controller import parse_capture_headers
from src.visual_servoing_dataset.utils.logger import console, logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Stream da câmera do smartphone via Cam-App (exclusivamente em modo 'auto')."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Caminho do arquivo de configuração para ler o IP do smartphone (padrão: config.yaml).",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Sobrescrever o IP do smartphone manualmente (ex: 192.168.158.65).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Sobrescrever a porta HTTP do Cam-App (padrão: 8080).",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="Largura inicial da janela redimensionável (padrão: 1280).",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=720,
        help="Altura inicial da janela redimensionável (padrão: 720).",
    )
    parser.add_argument(
        "--charuco",
        action="store_true",
        help="Iniciar com o overlay de detecção ChArUco (ID=8 + perímetros) ativado.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Iniciar sem a mira central, ROI e grade de enquadramento.",
    )
    parser.add_argument(
        "--roi",
        type=float,
        default=0.8,
        help="Fração da largura e altura para o retângulo de ROI centralizado (padrão: 0.8).",
    )
    return parser.parse_args()


def resolve_camera_endpoint(args: argparse.Namespace) -> Tuple[str, int, AppConfig]:
    """Obtém o IP e a porta do smartphone a partir dos argumentos CLI ou do config.yaml."""
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = load_config(cfg_path)
    else:
        cfg = AppConfig()

    ip = args.ip or (cfg.smartphone.ips[0] if cfg.smartphone.ips else "192.168.158.65")
    port = args.port or cfg.smartphone.port or 8080
    if ":" in ip:
        host_part, port_part = ip.split(":", 1)
        ip = host_part
        port = int(port_part)

    return ip, port, cfg


def apply_auto_mode(session: requests.Session, base_url: str, timeout: float = 3.0) -> bool:
    """Garante que a câmera do smartphone opere exclusivamente no modo 'auto'."""
    auto_payload = {
        "control_mode": "auto",
        "auto_mode": True,
        "focus_distance": 0.0,
        "iso": 100,
        "exposure_time_ns": 10000000,
        "ois_enabled": True,
        "eis_enabled": True,
    }
    try:
        resp = session.post(f"{base_url}/config", json=auto_payload, timeout=timeout)
        if resp.status_code == 200:
            logger.info("Modo 'auto' aplicado com sucesso no Cam-App.")
            return True
        logger.warning(f"POST /config retornou status {resp.status_code}: {resp.text}")
    except RequestException as exc:
        logger.warning(f"Não foi possível aplicar modo 'auto' em {base_url}/config: {exc}")
    return False


def _draw_dashed_line(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    dash_length: int = 12,
    gap_length: int = 8,
) -> None:
    """Desenha um segmento de linha pontilhada/tracejada entre pt1 e pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1.0:
        return
    ux = (x2 - x1) / dist
    uy = (y2 - y1) / dist
    step = dash_length + gap_length
    pos = 0.0
    while pos < dist:
        end_pos = min(pos + dash_length, dist)
        sx = int(round(x1 + ux * pos))
        sy = int(round(y1 + uy * pos))
        ex = int(round(x1 + ux * end_pos))
        ey = int(round(y1 + uy * end_pos))
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos += step


def draw_framing_grid(frame: np.ndarray, roi_fraction: float = 0.8) -> None:
    """Desenha mira central em (W/2, H/2), linhas de terços e o retângulo pontilhado vermelho do ROI."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Linhas de terços suaves
    for frac in (1 / 3, 2 / 3):
        x_line = int(round(w * frac))
        y_line = int(round(h * frac))
        cv2.line(frame, (x_line, 0), (x_line, h), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.line(frame, (0, y_line), (w, y_line), (90, 90, 90), 1, cv2.LINE_AA)

    # Retângulo pontilhado em vermelho indicando o ROI centralizado (padrão: 0.8 da largura e altura)
    roi_frac = float(np.clip(roi_fraction, 0.05, 1.0))
    half_roi_w = int(round((w * roi_frac) / 2.0))
    half_roi_h = int(round((h * roi_frac) / 2.0))
    x_min, x_max = cx - half_roi_w, cx + half_roi_w
    y_min, y_max = cy - half_roi_h, cy + half_roi_h

    red_bgr = (0, 0, 255)
    _draw_dashed_line(frame, (x_min, y_min), (x_max, y_min), color=red_bgr, thickness=2)
    _draw_dashed_line(frame, (x_max, y_min), (x_max, y_max), color=red_bgr, thickness=2)
    _draw_dashed_line(frame, (x_max, y_max), (x_min, y_max), color=red_bgr, thickness=2)
    _draw_dashed_line(frame, (x_min, y_max), (x_min, y_min), color=red_bgr, thickness=2)

    cv2.putText(
        frame,
        f"ROI {int(round(roi_frac * 100))}% ({x_max - x_min}x{y_max - y_min}px)",
        (x_min + 8, max(y_min - 8, 108)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        red_bgr,
        2,
        cv2.LINE_AA,
    )

    # Eixos centrais e mira no centro óptico nominal
    cv2.line(frame, (cx, 0), (cx, h), (180, 180, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (0, cy), (w, cy), (180, 180, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 18, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)


def draw_charuco_overlay(
    frame: np.ndarray,
    solver: CharucoCalibrationSolver,
    center_marker_id: int = 8,
) -> None:
    """Detecta e desenha em tempo real o centro do ArUco ID=8 e o balanço de perímetros (15,16 vs 0,1)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_det = cv2.aruco.ArucoDetector(solver.aruco_dict, solver._detector_params)
    corners, ids, _ = aruco_det.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    meas = solver.analyze_charuco_centering_and_parallelism(
        image_bgr=frame,
        center_marker_id=center_marker_id,
        debug_save_path=None,
    )
    if not meas.valid:
        cv2.putText(
            frame,
            f"ChArUco: {meas.rejection_reason}",
            (24, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 100, 255),
            2,
            cv2.LINE_AA,
        )
        return

    ic_pt = (int(round(meas.image_center_u_px)), int(round(meas.image_center_v_px)))
    c8_pt = (int(round(meas.center_u_px)), int(round(meas.center_v_px)))
    color_center = (0, 255, 0) if meas.center_error_norm_px <= 12.0 else (0, 165, 255)

    cv2.circle(frame, c8_pt, 9, color_center, -1, cv2.LINE_AA)
    cv2.arrowedLine(frame, c8_pt, ic_pt, color_center, 2, tipLength=0.15)

    banner = (
        f"ID={center_marker_id} Erro: {meas.center_error_norm_px:.1f}px ({meas.center_error_norm_mm:.1f}mm) | "
        f"Esq{meas.left_markers_used}:{meas.sum_left_perimeters_px:.0f} vs "
        f"Dir{meas.right_markers_used}:{meas.sum_right_perimeters_px:.0f}px ({meas.horiz_asymmetry_pct:+.1f}%) | "
        f"Topo/Base:{meas.vert_asymmetry_pct:+.1f}% | Roll:{meas.roll_error_deg:+.1f} deg"
    )
    cv2.rectangle(frame, (16, frame.shape[0] - 58), (frame.shape[1] - 16, frame.shape[0] - 14), (20, 20, 20), -1)
    cv2.putText(
        frame,
        banner,
        (26, frame.shape[0] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color_center,
        2,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    meta: Dict[str, Any],
    fps: float,
    latency_ms: float,
    ip: str,
    show_grid: bool,
    show_charuco: bool,
) -> None:
    """Desenha barra superior com telemetria do modo 'auto' e atalhos de teclado."""
    h, w = frame.shape[:2]
    iso = meta.get("sensor_sensitivity", meta.get("iso", "--"))
    exp_ms = meta.get("exposure_time_ms")
    if exp_ms is None and isinstance(meta.get("sensor_exposure_time"), (int, float)):
        exp_ms = meta["sensor_exposure_time"] / 1e6
    exp_str = f"{exp_ms:.2f} ms" if isinstance(exp_ms, (int, float)) else "--"

    focus_d = meta.get("lens_focus_distance", meta.get("focus_distance"))
    if isinstance(focus_d, (int, float)) and focus_d > 0.01:
        dist_cm = 100.0 / float(focus_d)
        focus_str = f"{focus_d:.2f} D (~{dist_cm:.1f} cm)"
    elif isinstance(focus_d, (int, float)):
        focus_str = f"{focus_d:.2f} D (inf)"
    else:
        focus_str = "--"

    # Fundo semitransparente no topo
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 12), (w - 14, 92), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    line1 = (
        f"[MODO: AUTO] {ip} ({w}x{h}) | FPS: {fps:.1f} ({latency_ms:.0f} ms) | "
        f"ISO: {iso} | Exp: {exp_str} | Foco: {focus_str}"
    )
    line2 = (
        f"Atalhos: [q] Sair | [g] Grade/Mira ({'ON' if show_grid else 'OFF'}) | "
        f"[c] ChArUco ({'ON' if show_charuco else 'OFF'}) | [r] Refocar Auto | [s] Snapshot | [h] HUD"
    )

    cv2.putText(frame, line1, (26, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (26, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (210, 210, 210), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    ip, port, cfg = resolve_camera_endpoint(args)
    base_url = f"http://{ip}:{port}"
    capture_url = f"{base_url}/capture"
    window_title = f"Cam-App Live View (AUTO) - {ip}:{port}"

    console.rule(f"[bold cyan]Cam-App Live Stream (Modo AUTO) — {base_url}[/bold cyan]")
    logger.info(
        "Atalhos na janela: [q/ESC] Sair | [g] Grade/Mira | [c] ChArUco ID=8 | [r] Refocar Auto | [s] Snapshot | [h] HUD"
    )

    charuco_spec = CharucoBoardSpec(
        squares_x=cfg.calibration.charuco_squares_x,
        squares_y=cfg.calibration.charuco_squares_y,
        square_length_m=cfg.calibration.charuco_square_length_m,
        marker_length_m=cfg.calibration.charuco_marker_length_m,
        dictionary=cfg.calibration.charuco_dictionary,
    )
    solver = CharucoCalibrationSolver(charuco_spec=charuco_spec)

    show_grid = not args.no_grid
    show_charuco = bool(args.charuco)
    show_hud = True

    session = requests.Session()
    apply_auto_mode(session, base_url)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, args.window_width, args.window_height)

    fps_ema: float = 0.0
    last_frame_time = time.perf_counter()
    last_valid_frame: Optional[np.ndarray] = None

    try:
        while True:
            req_start = time.perf_counter()
            try:
                response = session.get(capture_url, timeout=2.5)
                latency_ms = (time.perf_counter() - req_start) * 1000.0

                if response.status_code == 200 and response.content:
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if frame is not None and frame.size > 0:
                        last_valid_frame = frame.copy()
                        now = time.perf_counter()
                        inst_fps = 1.0 / max(now - last_frame_time, 1e-4)
                        last_frame_time = now
                        fps_ema = inst_fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)

                        meta = parse_capture_headers(response.headers)

                        if show_grid:
                            draw_framing_grid(frame, roi_fraction=args.roi)
                        if show_charuco:
                            draw_charuco_overlay(
                                frame,
                                solver=solver,
                                center_marker_id=cfg.calibration.center_marker_id,
                            )
                        if show_hud:
                            draw_hud(
                                frame=frame,
                                meta=meta,
                                fps=fps_ema,
                                latency_ms=latency_ms,
                                ip=ip,
                                show_grid=show_grid,
                                show_charuco=show_charuco,
                            )

                        cv2.imshow(window_title, frame)

            except RequestException as exc:
                logger.warning(f"Erro de conexão com {capture_url}: {exc}")
                time.sleep(0.3)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' ou ESC
                break
            elif key == ord("g"):
                show_grid = not show_grid
            elif key == ord("c"):
                show_charuco = not show_charuco
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("r"):
                apply_auto_mode(session, base_url)
            elif key == ord("s") and last_valid_frame is not None:
                snap_dir = Path("assets/snapshots")
                snap_dir.mkdir(parents=True, exist_ok=True)
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = snap_dir / f"snapshot_{ts_str}.jpg"
                cv2.imwrite(str(snap_path), last_valid_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                logger.info(f"Snapshot salvo em: {snap_path.resolve()}")

            # Encerrar se a janela for fechada no botão 'X'
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        logger.info("Stream interrompido pelo usuário.")
    finally:
        session.close()
        cv2.destroyAllWindows()
        logger.info("Janela encerrada.")


if __name__ == "__main__":
    main()
