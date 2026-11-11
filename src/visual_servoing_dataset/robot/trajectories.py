"""Compatibilidade com o módulo trajectories anterior, redirecionando para PoseFamilyGenerator em SE(3)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..config import BasePose, Joint, Pose, PoseSamplingConfig, ensure_pose
from .pose_generator import PoseFamilyGenerator


@dataclass
class PoseWaypoint:
    """Representa um waypoint de captura derivado de PlannedPoseCandidate."""
    id: str
    filename: str
    variation_type: str
    pose: Pose
    pose_family: str = "P0"
    pair_id: str = ""
    joint: Optional[Joint] = None
    delta_applied: Dict[str, float] = field(default_factory=dict)


class PoseVariationGenerator(PoseFamilyGenerator):
    """Adaptador sobre PoseFamilyGenerator mantendo método generate_waypoints_for_base_pose."""

    def generate_waypoints_for_base_pose(
        self,
        p0: Union[BasePose, Pose],
        scene_id: str = "scene_01",
    ) -> List[PoseWaypoint]:
        candidates, _ = self.generate_scene_poses(scene_id=scene_id, p0=p0)
        waypoints: List[PoseWaypoint] = []
        for c in candidates:
            xi = c.xi_local
            waypoints.append(
                PoseWaypoint(
                    id=c.pose_id,
                    filename=c.filename,
                    variation_type=c.variation_type,
                    pose=ensure_pose(c.target_cartesian_pose),
                    pose_family=c.pose_family,
                    pair_id=c.pair_id,
                    delta_applied={
                        "dx": xi[0],
                        "dy": xi[1],
                        "dz": xi[2],
                        "drx": xi[3],
                        "dry": xi[4],
                        "drz": xi[5],
                    },
                )
            )
        return waypoints
