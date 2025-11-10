#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Navigation task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Navigation task.

    A cooperative navigation task where agents must reach goal landmarks while avoiding
    collisions with each other. Can be configured with shared or individual goals and
    rewards. Tests coordination and collision avoidance.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents in the environment.
        collisions: Whether collisions between agents are penalized.
        agents_with_same_goal: Number of agents assigned to the same goal.
        observe_all_goals: Whether agents can observe all goals or just their own.
        shared_rew: Whether to use shared rewards across all agents.
        split_goals: Whether goals are split among agents.
        lidar_range: Range of lidar sensors for obstacle detection.
        agent_radius: Physical radius of agents.
    """

    max_steps: Any = MISSING
    n_agents: Any = MISSING
    collisions: Any = MISSING
    agents_with_same_goal: Any = MISSING
    observe_all_goals: Any = MISSING
    shared_rew: Any = MISSING
    split_goals: Any = MISSING
    lidar_range: Any = MISSING
    agent_radius: Any = MISSING
