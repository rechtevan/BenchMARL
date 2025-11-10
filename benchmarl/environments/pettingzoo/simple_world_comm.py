#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


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

    task: str = MISSING
    max_cycles: int = MISSING
    num_good: int = MISSING
    num_adversaries: int = MISSING
    num_obstacles: int = MISSING
    num_food: int = MISSING
    num_forests: int = MISSING
