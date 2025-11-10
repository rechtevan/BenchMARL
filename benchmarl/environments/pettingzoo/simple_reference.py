#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""PettingZoo Simple Reference task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo Simple Reference task.

    A communication task where one agent must navigate to a specific landmark based on
    communication from another agent. Tests referential communication and coordination.
    Supports both continuous and discrete actions.

    Attributes:
        task: The task name identifier.
        max_cycles: Maximum number of environment steps per episode.
        local_ratio: Ratio of local to global rewards (0=fully global, 1=fully local).
    """

    task: Any = MISSING
    max_cycles: Any = MISSING
    local_ratio: Any = MISSING
