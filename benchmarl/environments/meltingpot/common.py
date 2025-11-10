#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Common utilities and base classes for MeltingPot environment wrappers."""

import copy
from typing import Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import (
    DoubleToFloat,
    DTypeCastTransform,
    EnvBase,
    FlattenObservation,
    Transform,
)

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING


class MeltingPotClass(TaskClass):
    """Task class for MeltingPot multi-agent reinforcement learning environments.

    MeltingPot is a suite of test scenarios for multi-agent RL that focuses on social
    dilemmas and cooperation. It provides RGB observations and supports categorical actions.
    Environments include scenarios like Clean Up, Prisoners Dilemma, and various coordination games.

    The environment provides global state as RGB images and requires special transforms
    for handling visual observations and interaction inventories.
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
        from torchrl.envs.libs.meltingpot import MeltingpotEnv

        config = copy.deepcopy(self.config)

        return lambda: MeltingpotEnv(
            substrate=self.name.lower(),
            categorical_actions=True,
            device=device,
            **config,
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
        return self.config.get("max_steps", 100)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        """Return the mapping of agent groups.

        Args:
            env: The environment instance.

        Returns:
            Dictionary mapping group names to agent lists.
        """
        return env.group_map

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        """Get env transforms.

        Returns:
            The requested env transforms.
        """
        interaction_inventories_keys = [
            (group, "observation", "INTERACTION_INVENTORIES")
            for group in self.group_map(env).keys()
            if (group, "observation", "INTERACTION_INVENTORIES")
            in env.observation_spec.keys(True, True)
        ]
        return [DoubleToFloat()] + (
            [
                FlattenObservation(
                    in_keys=interaction_inventories_keys,
                    first_dim=-2,
                    last_dim=-1,
                )
            ]
            if len(interaction_inventories_keys)
            else []
        )

    def get_replay_buffer_transforms(self, env: EnvBase, group: str) -> List[Transform]:
        """Get replay buffer transforms.

        Returns:
            The requested replay buffer transforms.
        """
        return [
            DTypeCastTransform(
                dtype_in=torch.uint8,
                dtype_out=torch.float,
                in_keys=[
                    "RGB",
                    (group, "observation", "RGB"),
                    ("next", "RGB"),
                    ("next", group, "observation", "RGB"),
                ],
                in_keys_inv=[],
            )
        ]

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        """Return the state specification.

        Args:
            env: The environment instance.

        Returns:
            State specification for the environment, or None if not applicable.
        """
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            del observation_spec[group]
        if list(observation_spec.keys()) != ["RGB"]:
            raise ValueError(
                f"More than one global state key found in observation spec {observation_spec}."
            )
        return observation_spec

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
        observation_spec = env.observation_spec.clone()
        for group_key in list(observation_spec.keys()):
            if group_key not in self.group_map(env).keys():
                del observation_spec[group_key]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        """Perform info spec operation.

        Returns:
            Result of the operation.
        """
        observation_spec = env.observation_spec.clone()
        for group_key in list(observation_spec.keys()):
            if group_key not in self.group_map(env).keys():
                del observation_spec[group_key]
            else:
                group_obs_spec = observation_spec[group_key]["observation"]
                del group_obs_spec["RGB"]
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
        return "meltingpot"

    @staticmethod
    def render_callback(experiment, env: EnvBase, data: TensorDictBase):
        """Perform render callback operation.

        Returns:
            Result of the operation.
        """
        return data.get("RGB")


class MeltingPotTask(Task):
    """Enum for meltingpot tasks."""

    PREDATOR_PREY__ALLEY_HUNT = None
    CLEAN_UP = None
    COLLABORATIVE_COOKING__CIRCUIT = None
    FRUIT_MARKET__CONCENTRIC_RIVERS = None
    COLLABORATIVE_COOKING__FIGURE_EIGHT = None
    PAINTBALL__KING_OF_THE_HILL = None
    FACTORY_COMMONS__EITHER_OR = None
    PURE_COORDINATION_IN_THE_MATRIX__ARENA = None
    RUNNING_WITH_SCISSORS_IN_THE_MATRIX__REPEATED = None
    COLLABORATIVE_COOKING__CRAMPED = None
    RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ARENA = None
    PRISONERS_DILEMMA_IN_THE_MATRIX__REPEATED = None
    TERRITORY__OPEN = None
    STAG_HUNT_IN_THE_MATRIX__REPEATED = None
    CHICKEN_IN_THE_MATRIX__REPEATED = None
    GIFT_REFINEMENTS = None
    PURE_COORDINATION_IN_THE_MATRIX__REPEATED = None
    COLLABORATIVE_COOKING__FORCED = None
    RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__ARENA = None
    BACH_OR_STRAVINSKY_IN_THE_MATRIX__ARENA = None
    CHEMISTRY__TWO_METABOLIC_CYCLES_WITH_DISTRACTORS = None
    COMMONS_HARVEST__PARTNERSHIP = None
    PREDATOR_PREY__OPEN = None
    TERRITORY__ROOMS = None
    HIDDEN_AGENDA = None
    COOP_MINING = None
    DAYCARE = None
    PRISONERS_DILEMMA_IN_THE_MATRIX__ARENA = None
    TERRITORY__INSIDE_OUT = None
    BACH_OR_STRAVINSKY_IN_THE_MATRIX__REPEATED = None
    COMMONS_HARVEST__CLOSED = None
    CHEMISTRY__THREE_METABOLIC_CYCLES_WITH_PLENTIFUL_DISTRACTORS = None
    STAG_HUNT_IN_THE_MATRIX__ARENA = None
    PAINTBALL__CAPTURE_THE_FLAG = None
    COLLABORATIVE_COOKING__CROWDED = None
    ALLELOPATHIC_HARVEST__OPEN = None
    COLLABORATIVE_COOKING__RING = None
    COMMONS_HARVEST__OPEN = None
    COINS = None
    PREDATOR_PREY__ORCHARD = None
    PREDATOR_PREY__RANDOM_FOREST = None
    COLLABORATIVE_COOKING__ASYMMETRIC = None
    RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__REPEATED = None
    CHEMISTRY__THREE_METABOLIC_CYCLES = None
    RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ONE_SHOT = None
    CHEMISTRY__TWO_METABOLIC_CYCLES = None
    CHICKEN_IN_THE_MATRIX__ARENA = None
    BOAT_RACE__EIGHT_RACES = None
    EXTERNALITY_MUSHROOMS__DENSE = None

    @staticmethod
    def associated_class():
        """Perform associated class operation.

        Returns:
            Result of the operation.
        """
        return MeltingPotClass
