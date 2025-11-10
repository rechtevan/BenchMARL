#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


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

    max_steps: int = MISSING
    n_passages: int = MISSING
    fixed_passage: bool = MISSING
    random_start_angle: bool = MISSING
