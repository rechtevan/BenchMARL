#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Simple Adversary task.

    A multi-agent particle environment where good agents must reach landmarks while
    adversarial agents try to interfere. Tests competitive and cooperative behaviors
    in mixed-motive scenarios. VMAS implementation of the MPE Simple Adversary environment.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Total number of good agents.
        n_adversaries: Number of adversarial agents.
    """

    max_steps: int = MISSING
    n_agents: int = MISSING
    n_adversaries: int = MISSING
