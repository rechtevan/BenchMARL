#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""VMAS Dispersion task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


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

    max_steps: Any = MISSING
    n_agents: Any = MISSING
    n_food: Any = MISSING
    share_reward: Any = MISSING
    food_radius: Any = MISSING
    penalise_by_time: Any = MISSING
