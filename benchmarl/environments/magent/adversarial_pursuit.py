#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the MAgent2 Adversarial Pursuit task.

    This task simulates a pursuit-evasion scenario where predators chase prey in a grid environment.
    The environment supports discrete actions and uses a minimap representation.

    Attributes:
        map_size: Size of the grid map.
        minimap_mode: Whether to use minimap observation mode.
        tag_penalty: Penalty applied when a predator tags prey.
        max_cycles: Maximum number of environment steps per episode.
        extra_features: Whether to include additional features in observations.
    """

    map_size: int = MISSING
    minimap_mode: bool = MISSING
    tag_penalty: float = MISSING
    max_cycles: int = MISSING
    extra_features: bool = MISSING
