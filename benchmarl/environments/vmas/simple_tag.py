#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Simple Tag task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Simple Tag task.

    A predator-prey environment where good agents (prey) try to avoid adversarial agents
    (predators) in an environment with obstacles and landmarks. Tests competitive multi-agent
    behavior and pursuit-evasion strategies. VMAS implementation of the MPE Simple Tag.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        num_good_agents: Number of good agents (prey).
        num_adversaries: Number of adversarial agents (predators).
        num_landmarks: Number of landmarks in the environment.
        shape_agent_rew: Whether to use shaped rewards for good agents.
        shape_adversary_rew: Whether to use shaped rewards for adversaries.
        agents_share_rew: Whether good agents share rewards.
        adversaries_share_rew: Whether adversaries share rewards.
        observe_same_team: Whether agents observe teammates.
        observe_pos: Whether agents observe absolute positions.
        observe_vel: Whether agents observe velocities.
        bound: Boundary limit for the environment.
        respawn_at_catch: Whether prey respawn when caught.
    """

    max_steps: Any = MISSING
    num_good_agents: Any = MISSING
    num_adversaries: Any = MISSING
    num_landmarks: Any = MISSING
    shape_agent_rew: Any = MISSING
    shape_adversary_rew: Any = MISSING
    agents_share_rew: Any = MISSING
    adversaries_share_rew: Any = MISSING
    observe_same_team: Any = MISSING
    observe_pos: Any = MISSING
    observe_vel: Any = MISSING
    bound: Any = MISSING
    respawn_at_catch: Any = MISSING
