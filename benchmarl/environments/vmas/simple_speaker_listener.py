#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Simple Speaker Listener task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Simple Speaker Listener task.

    A communication task where a speaker observes the goal and communicates to a listener
    which landmark to reach. The speaker cannot move and the listener cannot observe the goal.
    Tests asymmetric communication. VMAS implementation of the MPE Simple Speaker Listener.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
    """

    max_steps: Any = MISSING
