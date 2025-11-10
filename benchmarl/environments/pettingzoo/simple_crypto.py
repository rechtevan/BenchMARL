#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""PettingZoo Simple Crypto task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo Simple Crypto task.

    A communication task where agents must learn to encrypt and decrypt messages.
    One agent observes a goal, encrypts it, and sends it to another agent who must
    decrypt it to reach the correct landmark. Supports both continuous and discrete actions.

    Attributes:
        task: The task name identifier.
        max_cycles: Maximum number of environment steps per episode.
    """

    task: Any = MISSING
    max_cycles: Any = MISSING
