#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Simple Spread task.

    A cooperative task where agents must spread out to cover all landmarks in the environment.
    Agents are penalized for collisions and rewarded for covering landmarks. Tests
    coordination and spatial distribution. VMAS implementation of the MPE Simple Spread.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        obs_agents: Whether agents can observe other agents.
        n_agents: Number of agents in the environment.
    """

    max_steps: int = MISSING
    obs_agents: bool = MISSING
    n_agents: int = MISSING
