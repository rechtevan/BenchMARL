#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Transport task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


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

    max_steps: Any = MISSING
    n_agents: Any = MISSING
    n_packages: Any = MISSING
    package_width: Any = MISSING
    package_length: Any = MISSING
    package_mass: Any = MISSING
