#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Common utilities and base classes for VMAS environment wrappers."""

import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING


class VmasClass(TaskClass):
    """Task class for Vectorized Multi-Agent System (VMAS) environments.

    VMAS is a vectorized 2D physics simulator for multi-agent reinforcement learning,
    designed for efficient parallel execution on GPU. Supports a wide variety of tasks
    including navigation, transport, flocking, and multi-agent particle environments (MPE).

    The environment supports both continuous and discrete actions, and can run thousands
    of parallel environments efficiently. Does not provide global state by default.
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
        return lambda: VmasEnv(
            scenario=self.name.lower(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

    def supports_continuous_actions(self) -> bool:
        """Check if environment supports continuous actions.

        Returns:
            True if continuous actions are supported, False otherwise.
        """
        return True

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
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        """Return the mapping of agent groups.

        Args:
            env: The environment instance.

        Returns:
            Dictionary mapping group names to agent lists.
        """
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        """Return the state specification.

        Args:
            env: The environment instance.

        Returns:
            State specification for the environment, or None if not applicable.
        """
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        """Perform action mask spec operation.

        Returns:
            Result of the operation.
        """
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        """Return the observation specification.

        Args:
            env: The environment instance.

        Returns:
            Observation specification for the environment.
        """
        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        """Perform info spec operation.

        Returns:
            Result of the operation.
        """
        info_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            del info_spec[(group, "observation")]
        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec
        else:
            return None

    def action_spec(self, env: EnvBase) -> Composite:
        """Return the action specification.

        Args:
            env: The environment instance.

        Returns:
            Action specification for the environment.
        """
        return env.full_action_spec_unbatched

    @staticmethod
    def env_name() -> str:
        """Return the environment name.

        Returns:
            Name of the environment.
        """
        return "vmas"


class VmasTask(Task):
    """Enum for VMAS tasks."""

    BALANCE = None
    SAMPLING = None
    NAVIGATION = None
    TRANSPORT = None
    REVERSE_TRANSPORT = None
    WHEEL = None
    DISPERSION = None
    MULTI_GIVE_WAY = None
    DROPOUT = None
    GIVE_WAY = None
    WIND_FLOCKING = None
    PASSAGE = None
    JOINT_PASSAGE = None
    JOINT_PASSAGE_SIZE = None
    BALL_PASSAGE = None
    BALL_TRAJECTORY = None
    BUZZ_WIRE = None
    FLOCKING = None
    DISCOVERY = None
    FOOTBALL = None
    SIMPLE_ADVERSARY = None
    SIMPLE_CRYPTO = None
    SIMPLE_PUSH = None
    SIMPLE_REFERENCE = None
    SIMPLE_SPEAKER_LISTENER = None
    SIMPLE_SPREAD = None
    SIMPLE_TAG = None
    SIMPLE_WORLD_COMM = None

    @staticmethod
    def associated_class():
        """Perform associated class operation.

        Returns:
            Result of the operation.
        """
        return VmasClass
