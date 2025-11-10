#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""VMAS Football task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Football task.

    A competitive football/soccer environment where blue and red teams compete to score goals.
    Supports various configurations including AI opponents, shooting mechanics, and dense rewards.
    Tests competitive multi-agent coordination and strategic team play.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        n_blue_agents: Number of agents on the blue team.
        n_red_agents: Number of agents on the red team.
        ai_red_agents: Whether red team uses AI-controlled agents.
        physically_different: Whether agents have different physical properties.
        spawn_in_formation: Whether agents spawn in organized formations.
        formation_agents_per_column: Number of agents per column in formation.
        randomise_formation_indices: Whether to randomize agent positions within formation.
        only_blue_formation: Whether only blue team uses formation spawning.
        formation_noise: Amount of noise added to formation positions.
        n_traj_points: Number of trajectory points for AI heuristic planning.
        ai_strength: Overall strength of AI opponent.
        ai_decision_strength: Decision-making strength of AI opponent.
        ai_precision_strength: Precision/accuracy strength of AI opponent.
        agent_size: Physical size of agents.
        goal_size: Size of the goals.
        goal_depth: Depth dimension of the goals.
        pitch_length: Length of the football pitch.
        pitch_width: Width of the football pitch.
        ball_mass: Mass of the ball.
        ball_size: Size/radius of the ball.
        u_multiplier: Multiplier for movement action forces.
        enable_shooting: Whether shooting action is enabled.
        u_rot_multiplier: Multiplier for rotation actions.
        u_shoot_multiplier: Multiplier for shooting force.
        shooting_radius: Maximum distance for shooting.
        shooting_angle: Maximum angle deviation for shooting.
        max_speed: Maximum speed for agents.
        ball_max_speed: Maximum speed for the ball.
        dense_reward: Whether to use dense reward shaping.
        pos_shaping_factor_ball_goal: Reward shaping factor for ball-to-goal distance.
        pos_shaping_factor_agent_ball: Reward shaping factor for agent-to-ball distance.
        distance_to_ball_trigger: Distance threshold for ball proximity rewards.
        scoring_reward: Reward for scoring a goal.
        observe_teammates: Whether agents observe teammate positions.
        observe_adversaries: Whether agents observe opponent positions.
        dict_obs: Whether to use dictionary-based observations.
    """

    max_steps: Any = MISSING

    # Agents config
    n_blue_agents: Any = MISSING
    n_red_agents: Any = MISSING
    ai_red_agents: Any = MISSING
    physically_different: Any = MISSING

    # Agent spawning
    spawn_in_formation: Any = MISSING
    formation_agents_per_column: Any = MISSING
    randomise_formation_indices: Any = MISSING
    only_blue_formation: Any = MISSING
    formation_noise: Any = MISSING

    # Opponent heuristic config
    n_traj_points: Any = MISSING
    ai_strength: Any = MISSING
    ai_decision_strength: Any = MISSING
    ai_precision_strength: Any = MISSING

    # Task sizes
    agent_size: Any = MISSING
    goal_size: Any = MISSING
    goal_depth: Any = MISSING
    pitch_length: Any = MISSING
    pitch_width: Any = MISSING
    ball_mass: Any = MISSING
    ball_size: Any = MISSING

    # Actions
    u_multiplier: Any = MISSING

    # Actions shooting
    enable_shooting: Any = MISSING
    u_rot_multiplier: Any = MISSING
    u_shoot_multiplier: Any = MISSING
    shooting_radius: Any = MISSING
    shooting_angle: Any = MISSING

    # Speeds
    max_speed: Any = MISSING
    ball_max_speed: Any = MISSING

    # Rewards
    dense_reward: Any = MISSING
    pos_shaping_factor_ball_goal: Any = MISSING
    pos_shaping_factor_agent_ball: Any = MISSING
    distance_to_ball_trigger: Any = MISSING
    scoring_reward: Any = MISSING

    # Observations
    observe_teammates: Any = MISSING
    observe_adversaries: Any = MISSING
    dict_obs: Any = MISSING
