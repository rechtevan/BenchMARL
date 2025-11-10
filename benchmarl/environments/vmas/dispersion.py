#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Dispersion task.

    A cooperative task where agents must collect food items while dispersing across
    the environment. Tests balance between exploration, food collection, and spatial
    distribution. Agents may be penalized for time taken to complete the task.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents in the environment.
        n_food: Number of food items to collect.
        share_reward: Whether to use shared rewards across all agents.
        food_radius: Radius/size of food items.
        penalise_by_time: Whether to penalize agents based on time taken.
    """

    max_steps: int = MISSING
    n_agents: int = MISSING
    n_food: int = MISSING
    share_reward: bool = MISSING
    food_radius: float = MISSING
    penalise_by_time: bool = MISSING
