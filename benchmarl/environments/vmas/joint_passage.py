#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Joint Passage task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


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

    max_steps: Any = MISSING
    n_passages: Any = MISSING
    fixed_passage: Any = MISSING
    joint_length: Any = MISSING
    random_start_angle: Any = MISSING
    random_goal_angle: Any = MISSING
    observe_joint_angle: Any = MISSING
    asym_package: Any = MISSING
    mass_ratio: Any = MISSING
    mass_position: Any = MISSING
