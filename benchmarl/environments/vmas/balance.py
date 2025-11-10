#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Balance task.

    A cooperative task where agents must balance and carry a package to a goal position.
    The package requires coordination to keep balanced while moving. Tests physical
    coordination and multi-agent manipulation.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents carrying the package.
        random_package_pos_on_line: Whether to randomize package position on the line.
        package_mass: Mass of the package to carry.
    """

    max_steps: int = MISSING
    n_agents: int = MISSING
    random_package_pos_on_line: bool = MISSING
    package_mass: float = MISSING
