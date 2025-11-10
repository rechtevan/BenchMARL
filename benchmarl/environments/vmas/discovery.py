#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""VMAS Discovery task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Discovery task.

    A cooperative exploration task where agents must discover and cover target locations.
    Multiple agents may need to simultaneously cover each target. Tests exploration,
    coordination, and efficient target coverage.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents in the environment.
        n_targets: Number of targets to discover and cover.
        lidar_range: Range of lidar sensors for target detection.
        covering_range: Distance within which an agent covers a target.
        agents_per_target: Number of agents required to simultaneously cover a target.
        targets_respawn: Whether targets respawn after being covered.
        shared_reward: Whether to use shared rewards across all agents.
    """

    max_steps: Any = MISSING
    n_agents: Any = MISSING
    n_targets: Any = MISSING
    lidar_range: Any = MISSING
    covering_range: Any = MISSING
    agents_per_target: Any = MISSING
    targets_respawn: Any = MISSING
    shared_reward: Any = MISSING
