#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Joint Passage Size task.

    A variant of Joint Passage where the size constraint adds additional difficulty.
    Agents must navigate through passages while maintaining joint constraints and
    considering size limitations. Tests precise coordination under multiple constraints.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_passages: Number of passages to navigate through.
        fixed_passage: Whether passage positions are fixed or randomized.
        random_start_angle: Whether to randomize initial agent angles.
        random_goal_angle: Whether to randomize goal angles.
        observe_joint_angle: Whether agents observe the joint angle.
    """

    max_steps: int = MISSING
    n_passages: int = MISSING
    fixed_passage: bool = MISSING
    random_start_angle: bool = MISSING
    random_goal_angle: bool = MISSING
    observe_joint_angle: bool = MISSING
