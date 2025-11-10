#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Give Way task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Give Way task.

    A cooperative task where two agents must coordinate to pass through a narrow passage.
    One agent must give way to allow the other to pass. Tests coordination, spatial
    reasoning, and turn-taking behavior.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        mirror_passage: Whether to use mirrored passage configurations.
        observe_rel_pos: Whether agents observe relative positions of other agents.
        done_on_completion: Whether episode terminates upon successful completion.
        final_reward: Bonus reward given upon successful task completion.
    """

    max_steps: Any = MISSING
    mirror_passage: Any = MISSING
    observe_rel_pos: Any = MISSING
    done_on_completion: Any = MISSING
    final_reward: Any = MISSING
