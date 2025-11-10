#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Wind Flocking task.

    A flocking task where agents must maintain formation and move together in a desired
    direction while dealing with wind disturbances. Tests coordinated movement, formation
    control, and robustness to environmental perturbations.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        dist_shaping_factor: Reward shaping factor for inter-agent distances.
        rot_shaping_factor: Reward shaping factor for agent rotations/headings.
        vel_shaping_factor: Reward shaping factor for agent velocities.
        pos_shaping_factor: Reward shaping factor for agent positions.
        energy_shaping_factor: Reward shaping factor for energy consumption.
        wind_shaping_factor: Reward shaping factor for wind resistance.
        wind: Magnitude of wind force affecting agents.
        cover_angle_tolerance: Tolerance angle for covering/shielding other agents.
        horizon: Planning horizon for reward calculations.
        observe_rel_pos: Whether agents observe relative positions of neighbors.
        observe_rel_vel: Whether agents observe relative velocities of neighbors.
        observe_pos: Whether agents observe absolute positions.
        desired_vel: Desired velocity magnitude for the flock.
    """

    max_steps: int = MISSING
    dist_shaping_factor: float = MISSING
    rot_shaping_factor: float = MISSING
    vel_shaping_factor: float = MISSING
    pos_shaping_factor: float = MISSING
    energy_shaping_factor: float = MISSING
    wind_shaping_factor: float = MISSING
    wind: float = MISSING
    cover_angle_tolerance: float = MISSING
    horizon: int = MISSING
    observe_rel_pos: bool = MISSING
    observe_rel_vel: bool = MISSING
    observe_pos: bool = MISSING
    desired_vel: float = MISSING
