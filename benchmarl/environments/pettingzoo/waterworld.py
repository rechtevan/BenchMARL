#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


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

    task: str = MISSING
    max_cycles: int = MISSING
    n_pursuers: int = MISSING
    n_evaders: int = MISSING
    n_poisons: int = MISSING
    n_obstacles: int = MISSING
    n_coop: int = MISSING
    n_sensors: int = MISSING
    sensor_range: float = MISSING
    radius: float = MISSING
    obstacle_radius: float = MISSING
    pursuer_max_accel: float = MISSING
    pursuer_speed: float = MISSING
    evader_speed: float = MISSING
    poison_speed: float = MISSING
    poison_reward: float = MISSING
    food_reward: float = MISSING
    encounter_reward: float = MISSING
    thrust_penalty: float = MISSING
    local_ratio: float = MISSING
    speed_features: bool = MISSING
