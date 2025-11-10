#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""VMAS Ball Trajectory task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Ball Trajectory task.

    A cooperative task where agents must guide a ball along a desired trajectory.
    Tests coordinated manipulation and trajectory following capabilities.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        joints: Whether the ball is connected with joints to agents.
    """

    max_steps: Any = MISSING
    joints: Any = MISSING
