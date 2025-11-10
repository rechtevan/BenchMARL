#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""VMAS Ball Passage task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Ball Passage task.

    A cooperative task where agents must guide a ball through narrow passages.
    Tests precise coordination and manipulation of shared objects through constrained spaces.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_passages: Number of passages the ball must navigate through.
        fixed_passage: Whether passage positions are fixed or randomized.
        random_start_angle: Whether to randomize the initial ball angle.
    """

    max_steps: Any = MISSING
    n_passages: Any = MISSING
    fixed_passage: Any = MISSING
    random_start_angle: Any = MISSING
