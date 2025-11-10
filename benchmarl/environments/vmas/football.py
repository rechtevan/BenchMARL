#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


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

    max_steps: int = MISSING

    # Agents config
    n_blue_agents: int = MISSING
    n_red_agents: int = MISSING
    ai_red_agents: bool = MISSING
    physically_different: bool = MISSING

    # Agent spawning
    spawn_in_formation: bool = MISSING
    formation_agents_per_column: int = MISSING
    randomise_formation_indices: bool = MISSING
    only_blue_formation: bool = MISSING
    formation_noise: float = MISSING

    # Opponent heuristic config
    n_traj_points: int = MISSING
    ai_strength: float = MISSING
    ai_decision_strength: float = MISSING
    ai_precision_strength: float = MISSING

    # Task sizes
    agent_size: float = MISSING
    goal_size: float = MISSING
    goal_depth: float = MISSING
    pitch_length: float = MISSING
    pitch_width: float = MISSING
    ball_mass: float = MISSING
    ball_size: float = MISSING

    # Actions
    u_multiplier: float = MISSING

    # Actions shooting
    enable_shooting: bool = MISSING
    u_rot_multiplier: float = MISSING
    u_shoot_multiplier: float = MISSING
    shooting_radius: float = MISSING
    shooting_angle: float = MISSING

    # Speeds
    max_speed: float = MISSING
    ball_max_speed: float = MISSING

    # Rewards
    dense_reward: bool = MISSING
    pos_shaping_factor_ball_goal: float = MISSING
    pos_shaping_factor_agent_ball: float = MISSING
    distance_to_ball_trigger: float = MISSING
    scoring_reward: float = MISSING

    # Observations
    observe_teammates: bool = MISSING
    observe_adversaries: bool = MISSING
    dict_obs: bool = MISSING
