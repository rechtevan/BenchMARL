#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""PettingZoo MultiWalker task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo MultiWalker task.

    MultiWalker is a cooperative environment where multiple bipedal walkers must coordinate
    to carry a package across rough terrain. Walkers are rewarded for forward progress and
    penalized for falling or terminating early.

    Attributes:
        task: The task name identifier.
        n_walkers: Number of walker agents.
        shared_reward: Whether to share rewards among all walkers.
        max_cycles: Maximum number of environment steps per episode.
        position_noise: Amount of noise added to position observations.
        angle_noise: Amount of noise added to angle observations.
        forward_reward: Reward multiplier for forward movement.
        fall_reward: Reward/penalty when a walker falls.
        terminate_reward: Reward/penalty upon early termination.
        terminate_on_fall: Whether episode terminates when any walker falls.
        remove_on_fall: Whether to remove fallen walkers from the environment.
        terrain_length: Length of the terrain to traverse.
    """

    task: Any = MISSING
    n_walkers: Any = MISSING
    shared_reward: Any = MISSING
    max_cycles: Any = MISSING
    position_noise: Any = MISSING
    angle_noise: Any = MISSING
    forward_reward: Any = MISSING
    fall_reward: Any = MISSING
    terminate_reward: Any = MISSING
    terminate_on_fall: Any = MISSING
    remove_on_fall: Any = MISSING
    terrain_length: Any = MISSING
