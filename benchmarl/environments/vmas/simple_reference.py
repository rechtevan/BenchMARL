#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Simple Reference task.

    A communication task where agents must use referential communication to coordinate
    reaching specific landmarks. Tests language-like communication and coordination.
    VMAS implementation of the MPE Simple Reference environment.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
    """

    max_steps: int = MISSING
