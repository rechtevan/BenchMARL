#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


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

    task: str = MISSING
    n_walkers: int = MISSING
    shared_reward: bool = MISSING
    max_cycles: int = MISSING
    position_noise: float = MISSING
    angle_noise: float = MISSING
    forward_reward: float = MISSING
    fall_reward: float = MISSING
    terminate_reward: float = MISSING
    terminate_on_fall: bool = MISSING
    remove_on_fall: bool = MISSING
    terrain_length: int = MISSING
