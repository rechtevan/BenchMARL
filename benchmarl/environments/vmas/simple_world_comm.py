#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Simple World Comm task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Simple World Comm task.

    A communication task in a complex world with obstacles, forests, and food. Agents must
    communicate and coordinate to navigate to food while avoiding obstacles and adversaries.
    Tests communication in environments with partial observability and rich structure.
    VMAS implementation of the MPE Simple World Comm.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        num_good_agents: Number of good agents.
        num_adversaries: Number of adversarial agents.
        num_landmarks: Number of landmarks in the environment.
        num_food: Number of food items to collect.
        num_forests: Number of forest areas (provide partial observability).
    """

    max_steps: Any = MISSING
    num_good_agents: Any = MISSING
    num_adversaries: Any = MISSING
    num_landmarks: Any = MISSING
    num_food: Any = MISSING
    num_forests: Any = MISSING
