#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""PettingZoo Simple Tag task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo Simple Tag task.

    A predator-prey environment where good agents (prey) try to avoid adversarial agents
    (predators) in an environment with obstacles. Predators are rewarded for tagging prey,
    while prey are rewarded for avoiding capture. Supports both continuous and discrete actions.

    Attributes:
        task: The task name identifier.
        num_good: Number of good agents (prey).
        num_adversaries: Number of adversarial agents (predators).
        num_obstacles: Number of obstacles in the environment.
        max_cycles: Maximum number of environment steps per episode.
    """

    task: Any = MISSING
    num_good: Any = MISSING
    num_adversaries: Any = MISSING
    num_obstacles: Any = MISSING
    max_cycles: Any = MISSING
