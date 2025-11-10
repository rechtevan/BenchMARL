#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""PettingZoo Waterworld task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the PettingZoo Waterworld task.

    A continuous control environment where pursuer agents must catch evader agents
    (food) while avoiding poison targets. Pursuers have limited sensors and must
    coordinate to successfully catch evaders. Supports continuous actions only.

    Attributes:
        task: The task name identifier.
        max_cycles: Maximum number of environment steps per episode.
        n_pursuers: Number of pursuer agents.
        n_evaders: Number of evader agents (food targets).
        n_poisons: Number of poison targets to avoid.
        n_obstacles: Number of static obstacles.
        n_coop: Number of pursuers required to catch an evader.
        n_sensors: Number of sensor rays per agent.
        sensor_range: Maximum range of agent sensors.
        radius: Radius of pursuer agents.
        obstacle_radius: Radius of obstacle objects.
        pursuer_max_accel: Maximum acceleration for pursuer agents.
        pursuer_speed: Maximum speed for pursuer agents.
        evader_speed: Speed of evader agents.
        poison_speed: Speed of poison targets.
        poison_reward: Reward/penalty for touching poison.
        food_reward: Reward for catching food (evaders).
        encounter_reward: Reward for encounters with evaders.
        thrust_penalty: Penalty per unit of acceleration/thrust.
        local_ratio: Ratio of local to global rewards (0=fully global, 1=fully local).
        speed_features: Whether to include speed in observations.
    """

    task: Any = MISSING
    max_cycles: Any = MISSING
    n_pursuers: Any = MISSING
    n_evaders: Any = MISSING
    n_poisons: Any = MISSING
    n_obstacles: Any = MISSING
    n_coop: Any = MISSING
    n_sensors: Any = MISSING
    sensor_range: Any = MISSING
    radius: Any = MISSING
    obstacle_radius: Any = MISSING
    pursuer_max_accel: Any = MISSING
    pursuer_speed: Any = MISSING
    evader_speed: Any = MISSING
    poison_speed: Any = MISSING
    poison_reward: Any = MISSING
    food_reward: Any = MISSING
    encounter_reward: Any = MISSING
    thrust_penalty: Any = MISSING
    local_ratio: Any = MISSING
    speed_features: Any = MISSING
