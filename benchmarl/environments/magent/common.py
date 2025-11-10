#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Common utilities and base classes for MAgent environment wrappers."""

import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooWrapper

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING


class MAgentClass(TaskClass):
    """Task class for MAgent2 environments.

    MAgent2 is a many-agent reinforcement learning platform that supports large-scale
    multi-agent games with complex interactions. This class provides integration for
    MAgent2 environments like adversarial pursuit, battle, and gather scenarios.

    The environment uses discrete actions and provides state information along with
    agent-specific observations. It supports heterogeneous agent groups and action masking.
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

        return lambda: PettingZooWrapper(
            env=self.__get_env(config),
            return_state=True,
            seed=seed,
            done_on_any=False,
            use_mask=False,
            device=device,
        )

    def __get_env(self, config) -> EnvBase:
        try:
            from magent2.environments import (  # type: ignore[import-not-found]  # Optional dependency
                adversarial_pursuit_v4,
                # battle_v4,
                # battlefield_v5,
                # combined_arms_v6,
                # gather_v5,
                # tiger_deer_v4
            )
        except ImportError:
            raise ImportError(
                "Module `magent2` not found, install it using `pip install magent2`"
            ) from None

        envs = {
            "ADVERSARIAL_PURSUIT": adversarial_pursuit_v4,
            # "BATTLE": battle_v4,
            # "BATTLEFIELD": battlefield_v5,
            # "COMBINED_ARMS": combined_arms_v6,
            # "GATHER": gather_v5,
            # "TIGER_DEER": tiger_deer_v4
        }
        if self.name not in envs:
            raise Exception(f"{self.name} is not an environment of MAgent2")
        return envs[self.name].parallel_env(**config, render_mode="rgb_array")

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

    def has_state(self) -> bool:
        """Check has state.

        Returns:
            Boolean indicating the result.
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
        return Composite({"state": env.observation_spec["state"].clone()})

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
        return "magent"


class MAgentTask(Task):
    """Enum for MAgent2 tasks."""

    ADVERSARIAL_PURSUIT = None
    # BATTLE = None
    # BATTLEFIELD = None
    # COMBINED_ARMS = None
    # GATHER = None
    # TIGER_DEER = None

    @staticmethod
    def associated_class():
        """Perform associated class operation.

        Returns:
            Result of the operation.
        """
        return MAgentClass
