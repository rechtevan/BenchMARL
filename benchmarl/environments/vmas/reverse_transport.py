#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Reverse Transport task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Reverse Transport task.

    A variant of the Transport task where agents must transport packages in reverse or
    under additional constraints. Tests coordination and manipulation under more challenging
    conditions than standard transport.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents in the environment.
        package_width: Width dimension of the package.
        package_length: Length dimension of the package.
        package_mass: Mass of the package to transport.
    """

    max_steps: Any = MISSING
    n_agents: Any = MISSING
    package_width: Any = MISSING
    package_length: Any = MISSING
    package_mass: Any = MISSING
