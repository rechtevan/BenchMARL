#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo Simple Push task.

    A cooperative task where an agent must push a landmark towards a goal position.
    The agent is rewarded for moving the landmark closer to the goal. Supports both
    continuous and discrete actions.

    Attributes:
        task: The task name identifier.
        max_cycles: Maximum number of environment steps per episode.
    """

    task: str = MISSING
    max_cycles: int = MISSING
