#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Wheel task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Wheel task.

    A cooperative task where agents must rotate a line (wheel) at a desired angular velocity.
    Tests coordination in applying forces to achieve rotational motion. Agents must balance
    their actions to maintain steady rotation.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents controlling the wheel.
        line_length: Length of the line/wheel to rotate.
        line_mass: Mass of the line/wheel.
        desired_velocity: Target angular velocity for the wheel.
    """

    max_steps: Any = MISSING
    n_agents: Any = MISSING
    line_length: Any = MISSING
    line_mass: Any = MISSING
    desired_velocity: Any = MISSING
