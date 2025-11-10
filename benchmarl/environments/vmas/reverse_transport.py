#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


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

    max_steps: int = MISSING
    n_agents: int = MISSING
    package_width: float = MISSING
    package_length: float = MISSING
    package_mass: float = MISSING
