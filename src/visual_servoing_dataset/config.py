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
    """Configuração do Tool Center Point (deslocamento da ferramenta/smartphone)."""
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
    ip_address: str = "192.168.0.1"
    workspace_name: str = "VS_WORKSPACE"
    control_name: str = "DENSO_VS050"
    speed: float = 20.0
    accel: float = 20.0
    decel: float = 20.0
    stabilization_delay_sec: float = 1.5
    tcp_offset: TCPOffset = field(default_factory=TCPOffset)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        tcp_data = data.get("tcp_offset", {})
        tcp_offset = TCPOffset(**tcp_data) if tcp_data else TCPOffset()
        return cls(
            ip_address=data.get("ip_address", "192.168.0.1"),
            workspace_name=data.get("workspace_name", "VS_WORKSPACE"),
            control_name=data.get("control_name", "DENSO_VS050"),
            speed=data.get("speed", 20.0),
            accel=data.get("accel", 20.0),
            decel=data.get("decel", 20.0),
            stabilization_delay_sec=data.get("stabilization_delay_sec", 1.5),
            tcp_offset=tcp_offset
        )


@dataclass
class SmartphoneConfig:
    """Configuração dos smartphones Android para captura via ADB."""
    serials: List[str] = field(default_factory=list)
    camera_names: Dict[str, str] = field(default_factory=dict)
    capture_delay_sec: float = 2.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SmartphoneConfig":
        return cls(
            serials=data.get("serials", []),
            camera_names=data.get("camera_names", {}),
            capture_delay_sec=data.get("capture_delay_sec", 2.0)
        )


@dataclass
class PoseVariationsConfig:
    """Variações de movimento a partir de cada pose base P0."""
    delta_trans_xy: Dict[str, float] = field(default_factory=lambda: {"dx": 20.0, "dy": 15.0})
    delta_rot_z: float = 15.0
    delta_scale_z: float = -30.0
    delta_combined: Dict[str, float] = field(default_factory=lambda: {
        "dx": 15.0, "dy": 10.0, "dz": -20.0, "drz": 10.0
    })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseVariationsConfig":
        return cls(
            delta_trans_xy=data.get("delta_trans_xy", {"dx": 20.0, "dy": 15.0}),
            delta_rot_z=data.get("delta_rot_z", 15.0),
            delta_scale_z=data.get("delta_scale_z", -30.0),
            delta_combined=data.get("delta_combined", {"dx": 15.0, "dy": 10.0, "dz": -20.0, "drz": 10.0})
        )


@dataclass
class BasePose:
    """Pose cartesiana e/ou de juntas de referência (P0)."""
    cartesian: List[float] = field(default_factory=lambda: [300.0, 0.0, 400.0, 180.0, 0.0, 0.0, 0])
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
            cartesian=data.get("cartesian", [300.0, 0.0, 400.0, 180.0, 0.0, 0.0, 0]),
            joints=data.get("joints")
        )


@dataclass
class SceneConfig:
    """Configuração de uma cena específica."""
    name: str = "scene_01"
    description: str = "Cenário de teste 01"
    p0_pose: BasePose = field(default_factory=BasePose)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneConfig":
        p0_data = data.get("p0_pose", {})
        p0_pose = BasePose.from_dict(p0_data) if p0_data else BasePose()
        return cls(
            name=data.get("name", "scene_01"),
            description=data.get("description", ""),
            p0_pose=p0_pose
        )


@dataclass
class DatasetConfig:
    """Configuração dos diretórios de saída e metadados."""
    output_dir: str = "assets/dataset"
    save_raw_photos: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        return cls(
            output_dir=data.get("output_dir", "assets/dataset"),
            save_raw_photos=data.get("save_raw_photos", True)
        )


@dataclass
class AppConfig:
    """Configuração global do sistema."""
    dry_run: bool = False
    robot: RobotConfig = field(default_factory=RobotConfig)
    smartphone: SmartphoneConfig = field(default_factory=SmartphoneConfig)
    variations: PoseVariationsConfig = field(default_factory=PoseVariationsConfig)
    scenes: List[SceneConfig] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        scenes_data = data.get("scenes", [])
        scenes = [SceneConfig.from_dict(s) for s in scenes_data] if scenes_data else [
            SceneConfig(name="scene_01", p0_pose=BasePose([300.0, 0.0, 400.0, 180.0, 0.0, 0.0, 0])),
            SceneConfig(name="scene_02", p0_pose=BasePose([350.0, 50.0, 420.0, 180.0, 0.0, 0.0, 0]))
        ]
        
        return cls(
            dry_run=data.get("dry_run", False),
            robot=RobotConfig.from_dict(data.get("robot", {})),
            smartphone=SmartphoneConfig.from_dict(data.get("smartphone", {})),
            variations=PoseVariationsConfig.from_dict(data.get("variations", {})),
            scenes=scenes,
            dataset=DatasetConfig.from_dict(data.get("dataset", {}))
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
