#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Joint Passage Size task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


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

    max_steps: Any = MISSING
    n_passages: Any = MISSING
    fixed_passage: Any = MISSING
    random_start_angle: Any = MISSING
    random_goal_angle: Any = MISSING
    observe_joint_angle: Any = MISSING
