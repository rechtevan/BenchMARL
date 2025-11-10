#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""PettingZoo Simple Adversary task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo Simple Adversary task.

    A multi-agent particle environment where good agents must reach landmarks while
    adversarial agents try to prevent them. Good agents are rewarded for reaching landmarks,
    adversaries for staying near them. Supports both continuous and discrete actions.

    Attributes:
        task: The task name identifier.
        N: Number of agents (both good and adversarial).
        max_cycles: Maximum number of environment steps per episode.
    """

    task: Any = MISSING
    N: Any = MISSING
    max_cycles: Any = MISSING
