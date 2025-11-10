#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Passage task.

    A cooperative task where agents must navigate through narrow passages to reach goals.
    Tests coordination in constrained spaces and efficient path planning. Rewards can be
    shared or individual.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_passages: Number of passages to navigate through.
        shared_reward: Whether to use shared rewards across all agents.
    """

    max_steps: int = MISSING
    n_passages: int = MISSING
    shared_reward: bool = MISSING
