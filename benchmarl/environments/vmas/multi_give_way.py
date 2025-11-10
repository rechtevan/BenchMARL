#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Multi Give Way task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Multi Give Way task.

    A multi-agent extension of Give Way where multiple pairs of agents must coordinate
    to pass through passages. Tests scalable coordination and collision avoidance with
    multiple agent pairs.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        agent_collision_penalty: Penalty for collisions between agents.
    """

    max_steps: Any = MISSING
    agent_collision_penalty: Any = MISSING
