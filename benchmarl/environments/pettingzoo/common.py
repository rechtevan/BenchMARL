#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Common utilities and base classes for PettingZoo environment wrappers."""

import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooEnv

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING


class PettingZooClass(TaskClass):
    """Task class for PettingZoo multi-agent environments.

    PettingZoo is a library of diverse multi-agent environments following the Gym API.
    This class provides integration for PettingZoo environments including MPE (Multi-Particle
    Environment) scenarios and SISL environments like MultiWalker and Waterworld.

    Supports both continuous and discrete actions depending on the specific environment.
    Many environments provide global state information and action masks for invalid actions.
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
        if self.supports_continuous_actions() and self.supports_discrete_actions():
            config.update({"continuous_actions": continuous_actions})
        return lambda: PettingZooEnv(
            categorical_actions=True,
            device=device,
            seed=seed,
            parallel=True,
            return_state=self.has_state(),
            render_mode="rgb_array",
            **config,
        )

    def supports_continuous_actions(self) -> bool:
        """Check if environment supports continuous actions.

        Returns:
            True if continuous actions are supported, False otherwise.
        """
        return self.name in {
            "MULTIWALKER",
            "WATERWORLD",
            "SIMPLE_ADVERSARY",
            "SIMPLE_CRYPTO",
            "SIMPLE_PUSH",
            "SIMPLE_REFERENCE",
            "SIMPLE_SPEAKER_LISTENER",
            "SIMPLE_SPREAD",
            "SIMPLE_TAG",
            "SIMPLE_WORLD_COMM",
        }

    def supports_discrete_actions(self) -> bool:
        """Check if environment supports discrete actions.

        Returns:
            True if discrete actions are supported, False otherwise.
        """
        return self.name in {
            "SIMPLE_ADVERSARY",
            "SIMPLE_CRYPTO",
            "SIMPLE_PUSH",
            "SIMPLE_REFERENCE",
            "SIMPLE_SPEAKER_LISTENER",
            "SIMPLE_SPREAD",
            "SIMPLE_TAG",
            "SIMPLE_WORLD_COMM",
        }

    def has_state(self) -> bool:
        """Check has state.

        Returns:
            Boolean indicating the result.
        """
        return self.name in {
            "SIMPLE_ADVERSARY",
            "SIMPLE_CRYPTO",
            "SIMPLE_PUSH",
            "SIMPLE_REFERENCE",
            "SIMPLE_SPEAKER_LISTENER",
            "SIMPLE_SPREAD",
            "SIMPLE_TAG",
            "SIMPLE_WORLD_COMM",
        }

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
        return self.config["max_cycles"]

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
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        """Perform action mask spec operation.

        Returns:
            Result of the operation.
        """
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "action_mask":
                    del group_obs_spec[key]
            if group_obs_spec.is_empty():
                del observation_spec[group]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        if observation_spec.is_empty():
            return None
        return observation_spec

    def observation_spec(self, env: EnvBase) -> Composite:
        """Return the observation specification.

        Args:
            env: The environment instance.

        Returns:
            Observation specification for the environment.
        """
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        """Perform info spec operation.

        Returns:
            Result of the operation.
        """
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
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
    def env_name() -> str:
        """Return the environment name.

        Returns:
            Name of the environment.
        """
        return "pettingzoo"


class PettingZooTask(Task):
    """Enum for PettingZoo tasks."""

    MULTIWALKER = None
    WATERWORLD = None
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
        return PettingZooClass
