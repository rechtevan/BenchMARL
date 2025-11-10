#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""VMAS Flocking task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Flocking task.

    A classic flocking task where agents must maintain cohesion while avoiding collisions.
    Agents are rewarded for staying close together as a flock while navigating obstacles.
    Tests emergent coordination and collective behavior.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents in the flock.
        n_obstacles: Number of obstacles to avoid.
        collision_reward: Reward/penalty for collisions with obstacles or other agents.
    """

    max_steps: Any = MISSING
    n_agents: Any = MISSING
    n_obstacles: Any = MISSING
    collision_reward: Any = MISSING
