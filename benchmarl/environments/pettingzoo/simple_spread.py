#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""PettingZoo Simple Spread task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo Simple Spread task.

    A cooperative task where agents must spread out to cover all landmarks in the environment.
    Agents are penalized for collisions with each other and rewarded for covering landmarks.
    Supports both continuous and discrete actions.

    Attributes:
        task: The task name identifier.
        max_cycles: Maximum number of environment steps per episode.
        local_ratio: Ratio of local to global rewards (0=fully global, 1=fully local).
        N: Number of agents.
    """

    task: Any = MISSING
    max_cycles: Any = MISSING
    local_ratio: Any = MISSING
    N: Any = MISSING
