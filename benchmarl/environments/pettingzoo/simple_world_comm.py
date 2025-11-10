#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""PettingZoo Simple World Comm task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo Simple World Comm task.

    A communication task in a more complex world with obstacles, forests, and food.
    Agents must communicate and coordinate to navigate to food while avoiding obstacles.
    Tests communication in environments with partial observability and rich structure.
    Supports both continuous and discrete actions.

    Attributes:
        task: The task name identifier.
        max_cycles: Maximum number of environment steps per episode.
        num_good: Number of good agents.
        num_adversaries: Number of adversarial agents.
        num_obstacles: Number of obstacles in the environment.
        num_food: Number of food items.
        num_forests: Number of forest areas (provide cover/partial observability).
    """

    task: Any = MISSING
    max_cycles: Any = MISSING
    num_good: Any = MISSING
    num_adversaries: Any = MISSING
    num_obstacles: Any = MISSING
    num_food: Any = MISSING
    num_forests: Any = MISSING
