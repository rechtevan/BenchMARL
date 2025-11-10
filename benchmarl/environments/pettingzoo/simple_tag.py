#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


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

    task: str = MISSING
    num_good: int = MISSING
    num_adversaries: int = MISSING
    num_obstacles: int = MISSING
    max_cycles: int = MISSING
