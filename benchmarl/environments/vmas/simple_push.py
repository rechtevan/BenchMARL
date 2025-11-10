#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Simple Push task.

    A cooperative task where an agent must push a movable landmark to a goal position.
    Tests manipulation and goal-directed behavior. VMAS implementation of the MPE
    Simple Push environment.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
    """

    max_steps: int = MISSING
