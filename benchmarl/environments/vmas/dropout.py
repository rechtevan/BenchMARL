#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Dropout task.

    A task testing robustness to agent failures where agents must complete objectives
    while some agents may "drop out" or become inactive. Tests coordination and
    adaptability when team composition changes. Energy coefficient controls energy costs.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents in the environment.
        energy_coeff: Coefficient for energy consumption penalties.
        start_same_point: Whether all agents start at the same initial position.
    """

    max_steps: int = MISSING
    n_agents: int = MISSING
    energy_coeff: float = MISSING
    start_same_point: bool = MISSING
