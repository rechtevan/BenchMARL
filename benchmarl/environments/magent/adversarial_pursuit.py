#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""MAgent Adversarial Pursuit task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


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

    map_size: Any = MISSING
    minimap_mode: Any = MISSING
    tag_penalty: Any = MISSING
    max_cycles: Any = MISSING
    extra_features: Any = MISSING
