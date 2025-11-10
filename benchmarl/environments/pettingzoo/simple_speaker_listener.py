#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""PettingZoo Simple Speaker Listener task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo Simple Speaker Listener task.

    A communication task where a speaker agent observes the goal and must communicate
    to a listener agent which landmark to navigate to. The speaker cannot move and the
    listener cannot see the goal. Supports both continuous and discrete actions.

    Attributes:
        task: The task name identifier.
        max_cycles: Maximum number of environment steps per episode.
    """

    task: Any = MISSING
    max_cycles: Any = MISSING
