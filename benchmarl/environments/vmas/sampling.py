#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    """Configuration for the VMAS Sampling task.

    A cooperative task where agents must sample from multiple Gaussian distributions
    in the environment to maximize reward. Tests exploration, coordination, and
    information gathering in continuous spaces.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_agents: Number of agents in the environment.
        shared_rew: Whether to use shared rewards across all agents.
        n_gaussians: Number of Gaussian distributions to sample from.
        lidar_range: Range of lidar sensors.
        cov: Covariance of the Gaussian distributions.
        collisions: Whether collisions between agents are enabled.
        spawn_same_pos: Whether agents spawn at the same initial position.
    """

    max_steps: int = MISSING
    n_agents: int = MISSING
    shared_rew: bool = MISSING
    n_gaussians: int = MISSING
    lidar_range: float = MISSING
    cov: float = MISSING
    collisions: bool = MISSING
    spawn_same_pos: bool = MISSING
