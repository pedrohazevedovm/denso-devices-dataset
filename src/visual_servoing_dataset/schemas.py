"""Esquemas de dados e serialização para o Protocolo Experimental em 3 Fases (SE(3) + LHS)."""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def compute_canonical_sha256(payload: Dict[str, Any]) -> str:
    """Calcula o hash SHA-256 canônico de um dicionário JSON ignorando o próprio campo plan_sha256."""
    filtered = {k: v for k, v in payload.items() if k != "plan_sha256"}
    canonical_bytes = json.dumps(filtered, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def compute_file_sha256(filepath: Any) -> str:
    """Calcula o hash SHA-256 dos bytes de um arquivo em disco."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class PlannedPoseCandidate:
    """Representa uma pose candidata gerada nas famílias P0..P7 em SE(3)."""
    pose_id: str                           # ex: "p0_reference_start", "p1_tx_m100", "p7_lhs_001"
    pair_id: str                           # ex: "scene_01__p7_lhs_001__r00"
    pose_family: str                       # "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"
    variation_type: str                    # ex: "reference_start", "trans_x", "rot_x_roll", "lhs_6dof", "reference_end"
    filename: str                          # ex: "p7_lhs_001.jpg"
    repeat_index: int
    xi_local: List[float]                  # [dx, dy, dz, drx, dry, drz] no referencial T_0
    delta_t_local_4x4: List[List[float]]   # Matriz 4x4 row-major Delta T_{T,i}
    target_cartesian_pose: List[float]     # [X, Y, Z, Rx, Ry, Rz, Fig] na base do DENSO
    target_matrix_4x4: List[List[float]]   # Matriz 4x4 row-major ^B T_{T,i}^{alvo}
    status: str = "pending_approval"       # "pending_approval", "approved", "rejected"
    axis_fraction: Optional[float] = None  # Fração em [-1, 1] para P1..P6
    lhs_sample_index: Optional[int] = None
    lhs_normalized_u: Optional[List[float]] = None
    rejection_reason: Optional[str] = None
    rejected_at: Optional[str] = None
    replaces_pose_id: Optional[str] = None
    replacement_seed: Optional[int] = None
    verified_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannedPoseCandidate":
        return cls(**data)


@dataclass
class LHSGenerationMetadata:
    """Metadados de rastreabilidade matemática do Latin Hypercube Sampling (P7)."""
    seed: int
    scipy_version: str
    optimization: Optional[str]
    n_samples: int
    raw_lower_bounds: List[float]
    raw_upper_bounds: List[float]
    shrinkage_factor: float
    effective_lower_bounds: List[float]
    effective_upper_bounds: List[float]
    normalized_samples_u: List[List[float]]
    physical_samples_xi: List[List[float]]
    replacement_rule: str = (
        "Quando uma pose P7 é rejeitada na Fase 2, seu registro permanece intacto com status='rejected' "
        "e rejection_reason preenchido. Uma amostra substituta é gerada via LatinHypercube(d=6) com "
        "seed determinística derivada (replacement_seed = base_seed + 1000 + replacement_index) e "
        "vinculada via replaces_pose_id."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LHSGenerationMetadata":
        return cls(**data)


@dataclass
class ScenePosesPlan:
    """Plano de poses P0..P7 para uma cena específica."""
    scene_id: str
    scene_category: str
    description: str
    euler_convention: str
    p0_cartesian: List[float]
    p0_joints: Optional[List[float]]
    p0_matrix_4x4: List[List[float]]
    lower_bounds: List[float]
    upper_bounds: List[float]
    isolated_fractions: List[float]
    calibrated_manual_settings: Dict[str, Any]
    lhs_metadata: LHSGenerationMetadata
    poses: List[PlannedPoseCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_category": self.scene_category,
            "description": self.description,
            "euler_convention": self.euler_convention,
            "p0_cartesian": self.p0_cartesian,
            "p0_joints": self.p0_joints,
            "p0_matrix_4x4": self.p0_matrix_4x4,
            "lower_bounds": self.lower_bounds,
            "upper_bounds": self.upper_bounds,
            "isolated_fractions": self.isolated_fractions,
            "calibrated_manual_settings": self.calibrated_manual_settings,
            "lhs_metadata": self.lhs_metadata.to_dict(),
            "poses": [p.to_dict() for p in self.poses],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenePosesPlan":
        return cls(
            scene_id=data["scene_id"],
            scene_category=data.get("scene_category", "planar_core"),
            description=data.get("description", ""),
            euler_convention=data.get("euler_convention", "xyz"),
            p0_cartesian=data["p0_cartesian"],
            p0_joints=data.get("p0_joints"),
            p0_matrix_4x4=data["p0_matrix_4x4"],
            lower_bounds=data["lower_bounds"],
            upper_bounds=data["upper_bounds"],
            isolated_fractions=data["isolated_fractions"],
            calibrated_manual_settings=data.get("calibrated_manual_settings", {}),
            lhs_metadata=LHSGenerationMetadata.from_dict(data["lhs_metadata"]),
            poses=[PlannedPoseCandidate.from_dict(p) for p in data.get("poses", [])],
        )


@dataclass
class PosesPlanDocument:
    """Documento raiz poses_plan.json gerado na Fase 2 e congelado para a Fase 3."""
    schema_version: str
    session_id: str
    generated_at: str
    status: str  # "pending_approval" | "frozen"
    euler_convention: str
    mount_id: str
    tool_definition_id: str
    tcp_offset: Dict[str, Any]
    tcp_matrix_flange_to_tcp_4x4: List[List[float]]
    scenes: Dict[str, ScenePosesPlan]
    verified_at: Optional[str] = None
    plan_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "generated_at": self.generated_at,
            "verified_at": self.verified_at,
            "status": self.status,
            "euler_convention": self.euler_convention,
            "mount_id": self.mount_id,
            "tool_definition_id": self.tool_definition_id,
            "tcp_offset": self.tcp_offset,
            "tcp_matrix_flange_to_tcp_4x4": self.tcp_matrix_flange_to_tcp_4x4,
            "scenes": {k: v.to_dict() for k, v in self.scenes.items()},
            "plan_sha256": self.plan_sha256,
        }
        return payload

    def freeze(self, verified_at_iso: str) -> str:
        """Congela o plano de poses e calcula o SHA-256 canônico."""
        self.status = "frozen"
        self.verified_at = verified_at_iso
        temp_dict = self.to_dict()
        self.plan_sha256 = compute_canonical_sha256(temp_dict)
        return self.plan_sha256

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PosesPlanDocument":
        scenes = {
            k: ScenePosesPlan.from_dict(v)
            for k, v in data.get("scenes", {}).items()
        }
        return cls(
            schema_version=data.get("schema_version", "2.0.0"),
            session_id=data.get("session_id", "session_001"),
            generated_at=data.get("generated_at", ""),
            status=data.get("status", "pending_approval"),
            euler_convention=data.get("euler_convention", "xyz"),
            mount_id=data.get("mount_id", "MOUNT_FLANGE_V1"),
            tool_definition_id=data.get("tool_definition_id", "SMARTPHONE_TOOL_V1"),
            tcp_offset=data.get("tcp_offset", {}),
            tcp_matrix_flange_to_tcp_4x4=data.get("tcp_matrix_flange_to_tcp_4x4", []),
            scenes=scenes,
            verified_at=data.get("verified_at"),
            plan_sha256=data.get("plan_sha256"),
        )


def audit_camera_parameters(
    preset_name: str,
    requested_settings: Dict[str, Any],
    effective_metadata: Dict[str, Any],
    focus_tol_diopters: float = 0.05,
    iso_tol: int = 5,
    exposure_tol_ns: int = 500_000,
) -> Dict[str, Any]:
    """Compara parâmetros solicitados em POST /config com os metadados efetivos de X-Camera-Metadata.

    Destaca explicitamente se foco, exposição, ISO ou OIS/EIS divergiram ou se algum campo retornou null.
    """
    is_manual = str(requested_settings.get("control_mode", preset_name)).lower() == "manual"
    null_fields: List[str] = []
    divergences: Dict[str, Dict[str, Any]] = {}

    critical_keys = [
        "sensor_exposure_time",
        "sensor_sensitivity",
        "sensor_timestamp",
        "lens_focus_distance",
        "lens_state",
        "control_ae_mode",
        "control_af_mode",
        "control_awb_mode",
        "lens_optical_stabilization_mode",
        "control_video_stabilization_mode",
    ]
    for k in critical_keys:
        if effective_metadata.get(k) is None:
            null_fields.append(k)

    req_focus = requested_settings.get("focus_distance")
    eff_focus = effective_metadata.get("lens_focus_distance")
    if is_manual and req_focus is not None and eff_focus is not None:
        if abs(float(req_focus) - float(eff_focus)) > focus_tol_diopters:
            divergences["focus_distance"] = {
                "requested": float(req_focus),
                "effective": float(eff_focus),
                "delta": round(float(eff_focus) - float(req_focus), 6),
            }

    req_iso = requested_settings.get("iso")
    eff_iso = effective_metadata.get("sensor_sensitivity")
    if is_manual and req_iso is not None and eff_iso is not None:
        if abs(int(req_iso) - int(eff_iso)) > iso_tol:
            divergences["iso"] = {
                "requested": int(req_iso),
                "effective": int(eff_iso),
                "delta": int(eff_iso) - int(req_iso),
            }

    req_exp = requested_settings.get("exposure_time_ns")
    eff_exp = effective_metadata.get("sensor_exposure_time")
    if is_manual and req_exp is not None and eff_exp is not None:
        if abs(int(req_exp) - int(eff_exp)) > exposure_tol_ns:
            divergences["exposure_time_ns"] = {
                "requested": int(req_exp),
                "effective": int(eff_exp),
                "delta_ns": int(eff_exp) - int(req_exp),
            }

    req_ois = 1 if requested_settings.get("ois_enabled") else 0
    eff_ois = effective_metadata.get("lens_optical_stabilization_mode")
    if eff_ois is not None and int(eff_ois) != req_ois:
        divergences["ois_enabled"] = {
            "requested_mode": req_ois,
            "effective_mode": int(eff_ois),
        }

    req_eis = 1 if requested_settings.get("eis_enabled") else 0
    eff_eis = effective_metadata.get("control_video_stabilization_mode")
    if eff_eis is not None and int(eff_eis) != req_eis:
        divergences["eis_enabled"] = {
            "requested_mode": req_eis,
            "effective_mode": int(eff_eis),
        }

    lens_moving = effective_metadata.get("lens_state") == 1
    if lens_moving:
        divergences["lens_state"] = {"expected": 0, "effective": 1}

    return {
        "compliant": len(divergences) == 0 and len(null_fields) == 0,
        "null_fields": null_fields,
        "divergences": divergences,
        "lens_moving": lens_moving,
    }


@dataclass
class CalibrationProfileDocument:
    """Esquema do artefato congelado calibration_profile.json gerado na Fase 0."""
    schema_version: str
    calibration_id: str
    timestamp_utc: str
    smartphone_id: str
    camera_physical_id: str
    mount_id: str
    tool_definition_id: str
    resolution_px: List[int]
    locked_camera_params: Dict[str, Any]
    charuco_spec: Dict[str, Any]
    intrinsics: Dict[str, Any]
    hand_eye: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationProfileDocument":
        required_keys = (
            "schema_version",
            "calibration_id",
            "timestamp_utc",
            "smartphone_id",
            "camera_physical_id",
            "mount_id",
            "tool_definition_id",
            "resolution_px",
            "locked_camera_params",
            "charuco_spec",
            "intrinsics",
            "hand_eye",
        )
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"calibration_profile.json inválido. Chaves obrigatórias ausentes: {missing}")

        return cls(
            schema_version=str(data["schema_version"]),
            calibration_id=str(data["calibration_id"]),
            timestamp_utc=str(data["timestamp_utc"]),
            smartphone_id=str(data["smartphone_id"]),
            camera_physical_id=str(data["camera_physical_id"]),
            mount_id=str(data["mount_id"]),
            tool_definition_id=str(data["tool_definition_id"]),
            resolution_px=[int(v) for v in data["resolution_px"]],
            locked_camera_params=dict(data["locked_camera_params"]),
            charuco_spec=dict(data["charuco_spec"]),
            intrinsics=dict(data["intrinsics"]),
            hand_eye=dict(data["hand_eye"]),
        )

