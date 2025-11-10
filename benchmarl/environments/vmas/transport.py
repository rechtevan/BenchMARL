#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Transport task.

    A cooperative task where agents must transport packages to goal positions.
    Multiple packages with configurable properties test coordination and multi-object
    manipulation. Agents must work together to push/carry packages to their destinations.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents in the environment.
        n_packages: Number of packages to transport.
        package_width: Width dimension of packages.
        package_length: Length dimension of packages.
        package_mass: Mass of each package.
    """

    max_steps: int = MISSING
    n_agents: int = MISSING
    n_packages: int = MISSING
    package_width: float = MISSING
    package_length: float = MISSING
    package_mass: float = MISSING
