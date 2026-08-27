from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
from ..config import BasePose, PoseVariationsConfig, Pose, Joint, ensure_pose, ensure_joint


@dataclass
class PoseWaypoint:
    """Representa um ponto de captura com seu objeto Pose do aether_rdk e metadados de variação."""
    id: str  # ex: "p0_reference", "p1_trans_xy", etc.
    filename: str  # ex: "p0_reference.jpg"
    variation_type: str  # ex: "reference", "translation_xy", "rotation_z", "scale_z", "combined"
    pose: Pose  # Objeto Pose nativo do aether_rdk
    joint: Optional[Joint] = None  # Objeto Joint nativo do aether_rdk (se disponível)
    delta_applied: Dict[str, float] = field(default_factory=dict)  # Metadados das variações aplicadas


class PoseVariationGenerator:
    """Gerador de waypoints baseados nos objetos Pose do aether_rdk."""

    def __init__(self, variations_config: PoseVariationsConfig):
        self.config = variations_config

    def generate_waypoints_for_base_pose(self, p0: Union[BasePose, Pose]) -> List[PoseWaypoint]:
        """Gera a sequência P0..P4 em instâncias do objeto Pose a partir da pose base P0."""
        base_pose: Pose = p0.to_pose() if isinstance(p0, BasePose) else ensure_pose(p0)
        x0, y0, z0 = base_pose.x, base_pose.y, base_pose.z
        rx0, ry0, rz0 = base_pose.rx, base_pose.ry, base_pose.rz
        fig0 = base_pose.fig

        waypoints: List[PoseWaypoint] = []

        # 0. P0 Reference
        p0_obj = Pose(x0, y0, z0, rx0, ry0, rz0, fig0)
        waypoints.append(
            PoseWaypoint(
                id="p0_reference",
                filename="p0_reference.jpg",
                variation_type="reference",
                pose=p0_obj,
                delta_applied={"dx": 0.0, "dy": 0.0, "dz": 0.0, "drz": 0.0}
            )
        )

        # 1. P1 TransXY (Translação apenas em X/Y)
        dx_p1 = float(self.config.delta_trans_xy.get("dx", 20.0))
        dy_p1 = float(self.config.delta_trans_xy.get("dy", 15.0))
        p1_obj = Pose(x0 + dx_p1, y0 + dy_p1, z0, rx0, ry0, rz0, fig0)
        waypoints.append(
            PoseWaypoint(
                id="p1_trans_xy",
                filename="p1_trans_xy.jpg",
                variation_type="translation_xy",
                pose=p1_obj,
                delta_applied={"dx": dx_p1, "dy": dy_p1, "dz": 0.0, "drz": 0.0}
            )
        )

        # 2. P2 RotZ (Rotação apenas em Z / J6)
        drz_p2 = float(self.config.delta_rot_z)
        p2_obj = Pose(x0, y0, z0, rx0, ry0, rz0 + drz_p2, fig0)
        waypoints.append(
            PoseWaypoint(
                id="p2_rot_z",
                filename="p2_rot_z.jpg",
                variation_type="rotation_z",
                pose=p2_obj,
                delta_applied={"dx": 0.0, "dy": 0.0, "dz": 0.0, "drz": drz_p2}
            )
        )

        # 3. P3 ScaleZ (Profundidade apenas em Z)
        dz_p3 = float(self.config.delta_scale_z)
        p3_obj = Pose(x0, y0, z0 + dz_p3, rx0, ry0, rz0, fig0)
        waypoints.append(
            PoseWaypoint(
                id="p3_scale_z",
                filename="p3_scale_z.jpg",
                variation_type="scale_z",
                pose=p3_obj,
                delta_applied={"dx": 0.0, "dy": 0.0, "dz": dz_p3, "drz": 0.0}
            )
        )

        # 4. P4 Combined (Translação, Profundidade e Rotação Combinadas)
        dx_p4 = float(self.config.delta_combined.get("dx", 15.0))
        dy_p4 = float(self.config.delta_combined.get("dy", 10.0))
        dz_p4 = float(self.config.delta_combined.get("dz", -20.0))
        drz_p4 = float(self.config.delta_combined.get("drz", 10.0))
        p4_obj = Pose(x0 + dx_p4, y0 + dy_p4, z0 + dz_p4, rx0, ry0, rz0 + drz_p4, fig0)
        waypoints.append(
            PoseWaypoint(
                id="p4_combined",
                filename="p4_combined.jpg",
                variation_type="combined",
                pose=p4_obj,
                delta_applied={"dx": dx_p4, "dy": dy_p4, "dz": dz_p4, "drz": drz_p4}
            )
        )

        return waypoints
