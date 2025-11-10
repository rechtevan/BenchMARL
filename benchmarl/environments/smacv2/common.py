#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Common utilities and base classes for SMACv2 environment wrappers."""

import copy
from typing import Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.smacv2 import SMACv2Env

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING


class Smacv2Class(TaskClass):
    """Task class for StarCraft Multi-Agent Challenge v2 (SMACv2) environments.

    SMACv2 provides StarCraft II-based multi-agent reinforcement learning scenarios where
    teams of units must cooperate to defeat enemy forces. Supports various unit compositions
    across Protoss, Terran, and Zerg races with different difficulty levels (balanced and
    asymmetric matchups).

    The environment provides global state information, agent observations with action masks,
    and tracks battle outcomes including win rate and episode limit statistics.
    """

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        """Get the environment creation function.

        Args:
            num_envs: Number of parallel environments.
            continuous_actions: Whether to use continuous actions.
            seed: Random seed for reproducibility.
            device: Device to create environment on.

        Returns:
            Function that creates the environment.
        """
        config = copy.deepcopy(self.config)
        return lambda: SMACv2Env(
            categorical_actions=True, seed=seed, device=device, **config
        )

    def supports_continuous_actions(self) -> bool:
        """Check if environment supports continuous actions.

        Returns:
            True if continuous actions are supported, False otherwise.
        """
        return False

    def supports_discrete_actions(self) -> bool:
        """Check if environment supports discrete actions.

        Returns:
            True if discrete actions are supported, False otherwise.
        """
        return True

    def has_render(self, env: EnvBase) -> bool:
        """Check if environment supports rendering.

        Args:
            env: The environment instance.

        Returns:
            True if rendering is supported, False otherwise.
        """
        return True

    def max_steps(self, env: EnvBase) -> int:
        """Return the maximum number of steps per episode.

        Args:
            env: The environment instance.

        Returns:
            Maximum steps per episode, or None if unlimited.
        """
        return env.episode_limit

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        """Return the mapping of agent groups.

        Args:
            env: The environment instance.

        Returns:
            Dictionary mapping group names to agent lists.
        """
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        """Return the state specification.

        Args:
            env: The environment instance.

        Returns:
            State specification for the environment, or None if not applicable.
        """
        observation_spec = env.observation_spec.clone()
        del observation_spec["info"]
        del observation_spec["agents"]
        return observation_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        """Perform action mask spec operation.

        Returns:
            Result of the operation.
        """
        observation_spec = env.observation_spec.clone()
        del observation_spec["info"]
        del observation_spec["state"]
        del observation_spec[("agents", "observation")]
        return observation_spec

    def observation_spec(self, env: EnvBase) -> Composite:
        """Return the observation specification.

        Args:
            env: The environment instance.

        Returns:
            Observation specification for the environment.
        """
        observation_spec = env.observation_spec.clone()
        del observation_spec["info"]
        del observation_spec["state"]
        del observation_spec[("agents", "action_mask")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        """Perform info spec operation.

        Returns:
            Result of the operation.
        """
        observation_spec = env.observation_spec.clone()
        del observation_spec["state"]
        del observation_spec["agents"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        """Return the action specification.

        Args:
            env: The environment instance.

        Returns:
            Action specification for the environment.
        """
        return env.full_action_spec

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        """Perform log info operation.

        Returns:
            Result of the operation.
        """
        done = batch.get(("next", "done")).squeeze(-1)
        return {
            "collection/info/win_rate": batch.get(("next", "info", "battle_won"))[done]
            .to(torch.float)
            .mean()
            .item(),
            "collection/info/episode_limit_rate": batch.get(
                ("next", "info", "episode_limit")
            )[done]
            .to(torch.float)
            .mean()
            .item(),
        }

    @staticmethod
    def env_name() -> str:
        """Return the environment name.

        Returns:
            Name of the environment.
        """
        return "smacv2"


class Smacv2Task(Task):
    """Enum for SMACv2 tasks."""

    PROTOSS_5_VS_5 = None
    PROTOSS_10_VS_10 = None
    PROTOSS_10_VS_11 = None
    PROTOSS_20_VS_20 = None
    PROTOSS_20_VS_23 = None
    TERRAN_5_VS_5 = None
    TERRAN_10_VS_10 = None
    TERRAN_10_VS_11 = None
    TERRAN_20_VS_20 = None
    TERRAN_20_VS_23 = None
    ZERG_5_VS_5 = None
    ZERG_10_VS_10 = None
    ZERG_10_VS_11 = None
    ZERG_20_VS_20 = None
    ZERG_20_VS_23 = None

    @staticmethod
    def associated_class():
        """Perform associated class operation.

        Returns:
            Result of the operation.
        """
        return Smacv2Class
