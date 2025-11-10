#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""VMAS Buzz Wire task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Buzz Wire task.

    A cooperative task inspired by the buzz wire game where agents must navigate
    through a path without touching boundaries. Tests precise coordination and
    collision avoidance under tight constraints.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        random_start_angle: Whether to randomize the initial agent angles.
        collision_reward: Reward/penalty for colliding with boundaries.
    """

    max_steps: Any = MISSING
    random_start_angle: Any = MISSING
    collision_reward: Any = MISSING
