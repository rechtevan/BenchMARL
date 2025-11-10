#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Joint Passage task.

    A cooperative task where agents connected by joints must navigate through passages.
    The joint connection constrains agent movements, requiring tight coordination.
    Tests constrained multi-agent manipulation and synchronized movement.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_passages: Number of passages to navigate through.
        fixed_passage: Whether passage positions are fixed or randomized.
        joint_length: Length of the joint connecting agents.
        random_start_angle: Whether to randomize initial agent angles.
        random_goal_angle: Whether to randomize goal angles.
        observe_joint_angle: Whether agents observe the joint angle.
        asym_package: Whether to use asymmetric package (mass distribution).
        mass_ratio: Ratio of mass distribution in asymmetric package.
        mass_position: Position of center of mass in asymmetric package.
    """

    max_steps: int = MISSING
    n_passages: int = MISSING
    fixed_passage: bool = MISSING
    joint_length: float = MISSING
    random_start_angle: bool = MISSING
    random_goal_angle: bool = MISSING
    observe_joint_angle: bool = MISSING
    asym_package: bool = MISSING
    mass_ratio: float = MISSING
    mass_position: float = MISSING
