import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml

try:
    from aether_rdk.datatypes.pose import Pose
    from aether_rdk.datatypes.joint import Joint
    AETHER_RDK_AVAILABLE = True
except ImportError:
    AETHER_RDK_AVAILABLE = False

    @dataclass
    class Pose:
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
        rx: float = 0.0
        ry: float = 0.0
        rz: float = 0.0
        fig: int = 0

        def to_list(self) -> List[float]:
            return [self.x, self.y, self.z, self.rx, self.ry, self.rz, float(self.fig)]

        @staticmethod
        def from_list(data: List[float]) -> "Pose":
            if len(data) == 6:
                return Pose(data[0], data[1], data[2], data[3], data[4], data[5], 0)
            elif len(data) >= 7:
                return Pose(data[0], data[1], data[2], data[3], data[4], data[5], int(data[6]))
            raise ValueError(f"Tamanho de lista inválido para Pose: {len(data)}")

    @dataclass
    class Joint:
        joint_1: float = 0.0
        joint_2: float = 0.0
        joint_3: float = 0.0
        joint_4: float = 0.0
        joint_5: float = 0.0
        joint_6: float = 0.0

        def to_list(self) -> List[float]:
            return [self.joint_1, self.joint_2, self.joint_3, self.joint_4, self.joint_5, self.joint_6]

        @staticmethod
        def from_list(data: List[float]) -> "Joint":
            return Joint(data[0], data[1], data[2], data[3], data[4], data[5])


def ensure_pose(val: Union[Pose, List[float], Dict[str, float]]) -> Pose:
    """Converte lista, dict ou objeto existente para uma instância de Pose do aether_rdk."""
    if isinstance(val, Pose):
        return val
    elif isinstance(val, (list, tuple)):
        if len(val) == 6:
            return Pose(float(val[0]), float(val[1]), float(val[2]), float(val[3]), float(val[4]), float(val[5]), 0)
        elif len(val) >= 7:
            return Pose(float(val[0]), float(val[1]), float(val[2]), float(val[3]), float(val[4]), float(val[5]), int(val[6]))
        else:
            raise ValueError(f"Tamanho de lista inválido para Pose: {len(val)}")
    elif isinstance(val, dict):
        return Pose(
            float(val.get("x", 0.0)),
            float(val.get("y", 0.0)),
            float(val.get("z", 0.0)),
            float(val.get("rx", 0.0)),
            float(val.get("ry", 0.0)),
            float(val.get("rz", 0.0)),
            int(val.get("fig", 0))
        )
    else:
        raise TypeError(f"Tipo incompatível para Pose: {type(val)}")


def ensure_joint(val: Union[Joint, List[float], Dict[str, float]]) -> Joint:
    """Converte lista, dict ou objeto existente para uma instância de Joint do aether_rdk."""
    if isinstance(val, Joint):
        return val
    elif isinstance(val, (list, tuple)):
        if len(val) >= 6:
            return Joint(float(val[0]), float(val[1]), float(val[2]), float(val[3]), float(val[4]), float(val[5]))
        else:
            raise ValueError(f"Tamanho de lista inválido para Joint: {len(val)}")
    elif isinstance(val, dict):
        return Joint(
            float(val.get("joint_1", val.get("j1", 0.0))),
            float(val.get("joint_2", val.get("j2", 0.0))),
            float(val.get("joint_3", val.get("j3", 0.0))),
            float(val.get("joint_4", val.get("j4", 0.0))),
            float(val.get("joint_5", val.get("j5", 0.0))),
            float(val.get("joint_6", val.get("j6", 0.0)))
        )
    else:
        raise TypeError(f"Tipo incompatível para Joint: {type(val)}")


@dataclass
class TCPOffset:
    """Configuração do Tool Center Point (^F T_T: deslocamento do flange para o referencial da câmera/smartphone)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    tag: str = "SMARTPHONE_TOOL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "rx": self.rx, "ry": self.ry, "rz": self.rz,
            "tag": self.tag
        }


@dataclass
class RobotConfig:
    """Configuração do robô DENSO VS-050."""
    ip_address: str = "192.168.160.227"
    workspace_name: str = "VS_WORKSPACE"
    control_name: str = "DENSO_VS050"
    speed: float = 20.0
    accel: float = 20.0
    decel: float = 20.0
    robot_settling_time_sec: float = 1.5
    euler_convention: str = "xyz"  # Convenção DENSO RC8: 'xyz' extrínseco (equivalente a 'ZYX' intrínseco)
    mount_id: str = "MOUNT_FLANGE_V1"
    tool_definition_id: str = "SMARTPHONE_TOOL_V1"
    tcp_offset: TCPOffset = field(default_factory=TCPOffset)

    @property
    def stabilization_delay_sec(self) -> float:
        return self.robot_settling_time_sec

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        tcp_data = data.get("tcp_offset", {})
        tcp_offset = TCPOffset(**tcp_data) if tcp_data else TCPOffset()
        settling = data.get("robot_settling_time_sec", data.get("stabilization_delay_sec", 1.5))
        return cls(
            ip_address=data.get("ip_address", "192.168.160.227"),
            workspace_name=data.get("workspace_name", "VS_WORKSPACE"),
            control_name=data.get("control_name", "DENSO_VS050"),
            speed=float(data.get("speed", 20.0)),
            accel=float(data.get("accel", 20.0)),
            decel=float(data.get("decel", 20.0)),
            robot_settling_time_sec=float(settling),
            euler_convention=str(data.get("euler_convention", "xyz")),
            mount_id=str(data.get("mount_id", "MOUNT_FLANGE_V1")),
            tool_definition_id=str(data.get("tool_definition_id", tcp_offset.tag or "SMARTPHONE_TOOL_V1")),
            tcp_offset=tcp_offset,
        )


MANUAL_CALIBRADO_DESCRIPTION = (
    "Exposição, sensibilidade, foco e balanço de branco fixos, com controles de "
    "pós-processamento e estabilização desativáveis configurados para OFF."
)

AUTO_MODE_DESCRIPTION = (
    "Rotinas 3A do Android (AF, AE, AWB), controles de pós-processamento ISP "
    "e estabilização óptica/eletrônica (OIS/EIS) ativados."
)

DEFAULT_CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
    "auto": {
        "control_mode": "auto",
        "description": AUTO_MODE_DESCRIPTION,
        "focus_distance": 0.0,
        "iso": 100,
        "exposure_time_ns": 10000000,
        "ois_enabled": True,
        "eis_enabled": True,
        "auto_calibrate": False,
    },
    "manual_calibrado": {
        "control_mode": "manual",
        "description": MANUAL_CALIBRADO_DESCRIPTION,
        "focus_distance": 1.8621974,
        "iso": 1141,
        "exposure_time_ns": 16666660,
        "ois_enabled": False,
        "eis_enabled": False,
        "auto_calibrate": True,
    },
}


@dataclass
class SmartphoneConfig:
    """Configuração dos smartphones Android utilizando o aplicativo Cam-App (Camera2 HTTP Server)."""
    ips: List[str] = field(default_factory=lambda: ["192.168.155.44"])
    port: int = 8080
    smartphone_id: str = "moto_razr_60_ultra"
    camera_physical_id: str = "0"
    presets: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_CAMERA_PRESETS)
    )
    active_presets: List[str] = field(
        default_factory=lambda: ["auto", "manual_calibrado"]
    )
    auto_calibrate_manual: bool = True
    auto_3a_settle_sec: float = 1.0      # Tempo de acomodação de 3A (AF/AE/AWB) após ativar modo 'auto'
    manual_settle_sec: float = 1.0       # Tempo de estabilização da lente/sensor após ativar 'manual_calibrado'
    stabilization_delay_sec: float = 1.0
    timeout_sec: float = 3.0
    max_retries: int = 4
    retry_backoff_sec: float = 0.5
    camera_names: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SmartphoneConfig":
        ips = data.get("ips", ["192.168.155.44"])
        presets = copy.deepcopy(DEFAULT_CAMERA_PRESETS)

        user_presets = data.get("presets")
        if user_presets and isinstance(user_presets, dict):
            for key in ("auto", "manual_calibrado"):
                if key in user_presets and isinstance(user_presets[key], dict):
                    presets[key].update(user_presets[key])

        active_presets = data.get("active_presets", ["auto", "manual_calibrado"])
        if not isinstance(active_presets, list) or not active_presets:
            active_presets = ["auto", "manual_calibrado"]

        auto_calibrate_manual = bool(
            data.get(
                "auto_calibrate_manual",
                presets.get("manual_calibrado", {}).get("auto_calibrate", True),
            )
        )
        presets["manual_calibrado"]["auto_calibrate"] = auto_calibrate_manual

        base_settle = float(data.get("stabilization_delay_sec", 1.0))
        auto_settle = float(data.get("auto_3a_settle_sec", base_settle))
        manual_settle = float(data.get("manual_settle_sec", base_settle))

        return cls(
            ips=ips,
            port=int(data.get("port", 8080)),
            smartphone_id=str(data.get("smartphone_id", "moto_razr_60_ultra")),
            camera_physical_id=str(data.get("camera_physical_id", "0")),
            presets=presets,
            active_presets=active_presets,
            auto_calibrate_manual=auto_calibrate_manual,
            auto_3a_settle_sec=auto_settle,
            manual_settle_sec=manual_settle,
            stabilization_delay_sec=base_settle,
            timeout_sec=float(data.get("timeout_sec", 3.0)),
            max_retries=int(data.get("max_retries", 4)),
            retry_backoff_sec=float(data.get("retry_backoff_sec", 0.5)),
            camera_names=data.get("camera_names", {}),
        )


DEFAULT_LOWER_BOUNDS: List[float] = [-30.0, -25.0, -40.0, -10.0, -10.0, -15.0]
DEFAULT_UPPER_BOUNDS: List[float] = [30.0, 25.0, 40.0, 10.0, 10.0, 15.0]
DEFAULT_ISOLATED_FRACTIONS: List[float] = [-1.0, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.0]


@dataclass
class PoseSamplingConfig:
    """Configuração de geração das famílias de poses P0..P7 em SE(3) e amostragem LHS."""
    lower_bounds: List[float] = field(default_factory=lambda: list(DEFAULT_LOWER_BOUNDS))
    upper_bounds: List[float] = field(default_factory=lambda: list(DEFAULT_UPPER_BOUNDS))
    isolated_fractions: List[float] = field(default_factory=lambda: list(DEFAULT_ISOLATED_FRACTIONS))
    lhs_samples: int = 30
    lhs_shrinkage_factor: float = 0.75
    lhs_seed: int = 42
    lhs_optimization: Optional[str] = "random-cd"
    capture_p0_start: bool = True
    capture_p0_end: bool = True
    repeat_count: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseSamplingConfig":
        lower = data.get("lower_bounds", DEFAULT_LOWER_BOUNDS)
        upper = data.get("upper_bounds", DEFAULT_UPPER_BOUNDS)
        if len(lower) != 6 or len(upper) != 6:
            raise ValueError("lower_bounds e upper_bounds devem conter exatamente 6 elementos [x, y, z, rx, ry, rz].")

        fractions = data.get("isolated_fractions", DEFAULT_ISOLATED_FRACTIONS)
        return cls(
            lower_bounds=[float(v) for v in lower],
            upper_bounds=[float(v) for v in upper],
            isolated_fractions=[float(f) for f in fractions],
            lhs_samples=int(data.get("lhs_samples", 30)),
            lhs_shrinkage_factor=float(data.get("lhs_shrinkage_factor", 0.75)),
            lhs_seed=int(data.get("lhs_seed", 42)),
            lhs_optimization=data.get("lhs_optimization", "random-cd"),
            capture_p0_start=bool(data.get("capture_p0_start", True)),
            capture_p0_end=bool(data.get("capture_p0_end", True)),
            repeat_count=int(data.get("repeat_count", 1)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "isolated_fractions": list(self.isolated_fractions),
            "lhs_samples": self.lhs_samples,
            "lhs_shrinkage_factor": self.lhs_shrinkage_factor,
            "lhs_seed": self.lhs_seed,
            "lhs_optimization": self.lhs_optimization,
            "capture_p0_start": self.capture_p0_start,
            "capture_p0_end": self.capture_p0_end,
            "repeat_count": self.repeat_count,
        }


# Alias para compatibilidade com referências existentes
PoseVariationsConfig = PoseSamplingConfig


@dataclass
class BasePose:
    """Pose cartesiana e/ou de juntas de referência (P0)."""
    cartesian: List[float] = field(default_factory=lambda: [440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1])
    joints: Optional[List[float]] = None

    def to_pose(self) -> Pose:
        """Converte para objeto Pose do aether_rdk."""
        return ensure_pose(self.cartesian)

    def to_joint(self) -> Optional[Joint]:
        """Converte para objeto Joint do aether_rdk se juntas forem especificadas."""
        if self.joints:
            return ensure_joint(self.joints)
        return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasePose":
        return cls(
            cartesian=data.get("cartesian", [440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1]),
            joints=data.get("joints")
        )


@dataclass
class SceneConfig:
    """Configuração de uma cena experimental."""
    name: str = "scene_01"
    scene_category: str = "planar_core"  # "planar_core" vs "model_violation"
    description: str = "Cenário experimental 01"
    p0_pose: BasePose = field(default_factory=BasePose)
    camera_presets: List[str] = field(default_factory=lambda: ["auto", "manual_calibrado"])
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None

    @property
    def scene_id(self) -> str:
        return self.name

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneConfig":
        p0_data = data.get("p0_pose", {})
        p0_pose = BasePose.from_dict(p0_data) if p0_data else BasePose()

        presets = data.get("camera_presets")
        if not presets:
            single_preset = data.get("camera_preset")
            if single_preset:
                presets = [single_preset]
            else:
                presets = ["auto", "manual_calibrado"]

        lower = data.get("lower_bounds")
        upper = data.get("upper_bounds")

        return cls(
            name=str(data.get("name", data.get("scene_id", "scene_01"))),
            scene_category=str(data.get("scene_category", "planar_core")),
            description=str(data.get("description", "")),
            p0_pose=p0_pose,
            camera_presets=presets,
            lower_bounds=[float(v) for v in lower] if lower else None,
            upper_bounds=[float(v) for v in upper] if upper else None,
        )


@dataclass
class CalibrationConfig:
    """Configuração da Fase 0: Calibração Automatizada '2 em 1' (Intrínsecos + Distorção + Hand-Eye via ChArUco)."""
    calibration_id: str = "calib_phone01_mount01_20261110"
    calibration_file: Optional[str] = None
    output_dir: str = "assets/dataset/calibrations"
    rectify_on_collect: bool = True
    p_cal_0_pose: BasePose = field(
        default_factory=lambda: BasePose([440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1])
    )
    d0_mm: float = 350.0
    inner_tilt_deg: float = 15.0
    outer_tilt_deg: float = 22.0
    outer_rolls_deg: List[float] = field(default_factory=lambda: [20.0, -20.0, 35.0, -35.0])
    outer_radii_factors: List[float] = field(default_factory=lambda: [0.85, 1.15])
    edge_tilt_deg: float = 8.0
    edge_offset_fraction: float = 0.12
    edge_radius_factor: float = 0.90
    calibration_settling_time_sec: float = 1.2
    min_charuco_corners: int = 8
    hand_eye_method: str = "PARK"
    num_distortion_coeffs: int = 5
    use_camera2_initial_guess: bool = False
    charuco_squares_x: int = 7
    charuco_squares_y: int = 5
    charuco_square_length_m: float = 0.030
    charuco_marker_length_m: float = 0.022
    charuco_dictionary: str = "DICT_4X4_50"
    auto_center_p0: bool = True
    center_marker_id: int = 8
    center_max_iterations: int = 5
    center_tolerance_px: float = 12.0
    parallelism_tolerance_pct: float = 1.5
    center_kp_translation: float = 0.75
    center_kp_rotation: float = 0.70
    center_align_roll: bool = True
    center_max_step_mm: float = 35.0
    center_max_step_deg: float = 5.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationConfig":
        p0_data = data.get("p_cal_0_pose", {})
        p_cal_0 = BasePose.from_dict(p0_data) if p0_data else BasePose([440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1])
        charuco_data = data.get("charuco_spec", {})
        return cls(
            calibration_id=str(data.get("calibration_id", "calib_phone01_mount01_20261110")),
            calibration_file=data.get("calibration_file"),
            output_dir=str(data.get("output_dir", "assets/dataset/calibrations")),
            rectify_on_collect=bool(data.get("rectify_on_collect", True)),
            p_cal_0_pose=p_cal_0,
            d0_mm=float(data.get("d0_mm", 350.0)),
            inner_tilt_deg=float(data.get("inner_tilt_deg", 15.0)),
            outer_tilt_deg=float(data.get("outer_tilt_deg", 22.0)),
            outer_rolls_deg=[float(v) for v in data.get("outer_rolls_deg", [20.0, -20.0, 35.0, -35.0])],
            outer_radii_factors=[float(v) for v in data.get("outer_radii_factors", [0.85, 1.15])],
            edge_tilt_deg=float(data.get("edge_tilt_deg", 8.0)),
            edge_offset_fraction=float(data.get("edge_offset_fraction", 0.12)),
            edge_radius_factor=float(data.get("edge_radius_factor", 0.90)),
            calibration_settling_time_sec=float(data.get("calibration_settling_time_sec", 1.2)),
            min_charuco_corners=int(data.get("min_charuco_corners", 8)),
            hand_eye_method=str(data.get("hand_eye_method", "PARK")),
            num_distortion_coeffs=int(data.get("num_distortion_coeffs", 5)),
            use_camera2_initial_guess=bool(data.get("use_camera2_initial_guess", False)),
            charuco_squares_x=int(charuco_data.get("squares_x", data.get("charuco_squares_x", 7))),
            charuco_squares_y=int(charuco_data.get("squares_y", data.get("charuco_squares_y", 5))),
            charuco_square_length_m=float(charuco_data.get("square_length_m", data.get("charuco_square_length_m", 0.030))),
            charuco_marker_length_m=float(charuco_data.get("marker_length_m", data.get("charuco_marker_length_m", 0.022))),
            charuco_dictionary=str(charuco_data.get("dictionary", data.get("charuco_dictionary", "DICT_4X4_50"))),
            auto_center_p0=bool(data.get("auto_center_p0", True)),
            center_marker_id=int(data.get("center_marker_id", 8)),
            center_max_iterations=int(data.get("center_max_iterations", 5)),
            center_tolerance_px=float(data.get("center_tolerance_px", 12.0)),
            parallelism_tolerance_pct=float(data.get("parallelism_tolerance_pct", 1.5)),
            center_kp_translation=float(data.get("center_kp_translation", 0.75)),
            center_kp_rotation=float(data.get("center_kp_rotation", 0.70)),
            center_align_roll=bool(data.get("center_align_roll", True)),
            center_max_step_mm=float(data.get("center_max_step_mm", 35.0)),
            center_max_step_deg=float(data.get("center_max_step_deg", 5.0)),
        )


@dataclass
class DatasetConfig:
    """Configuração dos diretórios de saída, versão de esquema e arquivos de plano."""
    output_dir: str = "assets/dataset"
    session_id: str = "session_001"
    dataset_version: str = "2.0.0"
    schema_version: str = "2.0.0"
    session_plan_file: str = "session_plan.json"
    poses_plan_file: str = "poses_plan.json"
    save_raw_photos: bool = True
    save_tabular_csv: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        return cls(
            output_dir=str(data.get("output_dir", "assets/dataset")),
            session_id=str(data.get("session_id", "session_001")),
            dataset_version=str(data.get("dataset_version", "2.0.0")),
            schema_version=str(data.get("schema_version", "2.0.0")),
            session_plan_file=str(data.get("session_plan_file", "session_plan.json")),
            poses_plan_file=str(data.get("poses_plan_file", "poses_plan.json")),
            save_raw_photos=bool(data.get("save_raw_photos", True)),
            save_tabular_csv=bool(data.get("save_tabular_csv", True)),
        )


@dataclass
class AppConfig:
    """Configuração global do sistema."""
    dry_run: bool = False
    robot: RobotConfig = field(default_factory=RobotConfig)
    smartphone: SmartphoneConfig = field(default_factory=SmartphoneConfig)
    sampling: PoseSamplingConfig = field(default_factory=PoseSamplingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    scenes: List[SceneConfig] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    @property
    def variations(self) -> PoseSamplingConfig:
        """Alias para compatibilidade."""
        return self.sampling

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        scenes_data = data.get("scenes", [])
        scenes = [SceneConfig.from_dict(s) for s in scenes_data] if scenes_data else [
            SceneConfig(
                name="scene_01",
                scene_category="planar_core",
                description="Cenário planar principal",
                p0_pose=BasePose([440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1]),
            ),
            SceneConfig(
                name="scene_02",
                scene_category="model_violation",
                description="Cenário com relevo 3D / violação de planaridade",
                p0_pose=BasePose([440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1]),
            ),
        ]

        sampling_raw = data.get("sampling", data.get("variations", {}))
        return cls(
            dry_run=bool(data.get("dry_run", False)),
            robot=RobotConfig.from_dict(data.get("robot", {})),
            smartphone=SmartphoneConfig.from_dict(data.get("smartphone", {})),
            sampling=PoseSamplingConfig.from_dict(sampling_raw or {}),
            calibration=CalibrationConfig.from_dict(data.get("calibration", {})),
            scenes=scenes,
            dataset=DatasetConfig.from_dict(data.get("dataset", {})),
        )


def load_config(config_path: Union[str, Path]) -> AppConfig:
    """Carrega configuração de arquivo YAML ou JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Formato de arquivo de configuração não suportado: {path.suffix}")

    return AppConfig.from_dict(data or {})
