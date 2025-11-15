#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Ensemble algorithm wrapper for using different algorithms per agent group."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Type

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.objectives import LossModule

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig


class EnsembleAlgorithm(Algorithm):
    """Ensemble algorithm that uses different algorithms for different agent groups.

    This algorithm enables heterogeneous multi-agent learning by assigning
    a different algorithm to each agent group. Each group's algorithm operates
    independently with its own policies, critics, and training logic.

    The ensemble delegates all algorithm operations (loss computation, parameter
    retrieval, policy creation, batch processing) to the appropriate group-specific
    algorithm based on the agent group being processed.

    Args:
        algorithms_map: Dictionary mapping agent group names to their respective
            Algorithm instances.
        **kwargs: Additional arguments passed to the base Algorithm class.

    Attributes:
        algorithms_map: Dictionary storing the algorithm instance for each group.

    Example:
        Creating an ensemble with different algorithms per group::

            config = EnsembleAlgorithmConfig(
                algorithm_configs_map={
                    "group_1": MasacConfig(...),
                    "group_2": MappoConfig(...),
                }
            )
            algorithm = config.get_algorithm(experiment)
    """

    def __init__(self, algorithms_map, **kwargs):
        """Initialize the EnsembleAlgorithm instance.

        Parameters are documented in the class docstring.
        """
        super().__init__(**kwargs)
        self.algorithms_map = algorithms_map

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        return self.algorithms_map[group]._get_loss(group, policy_for_loss, continuous)

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return self.algorithms_map[group]._get_parameters(group, loss)

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        return self.algorithms_map[group]._get_policy_for_loss(
            group, model_config, continuous
        )

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        return self.algorithms_map[group]._get_policy_for_collection(
            policy_for_loss, group, continuous
        )

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        """Process and transform batch data using group-specific algorithm.

        Args:
            group: Name of the agent group.
            batch: Input batch to process.

        Returns:
            Processed batch with transformations applied by the group's algorithm.
        """
        return self.algorithms_map[group].process_batch(group, batch)

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        """Process loss values using group-specific algorithm.

        Args:
            group: Name of the agent group.
            loss_vals: Loss values to process.

        Returns:
            Processed loss values.
        """
        return self.algorithms_map[group].process_loss_vals(group, loss_vals)


@dataclass
class EnsembleAlgorithmConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.EnsembleAlgorithm`.

    This config enables heterogeneous multi-agent learning by assigning different
    algorithms to different agent groups. Each group can use its own learning
    algorithm with independent hyperparameters, policies, and critics.

    The ensemble algorithm can be either on-policy or off-policy, but all
    constituent algorithms must be consistent in their training paradigm.
    Similarly, action space support (continuous/discrete) is determined by
    the intersection of capabilities across all algorithms.

    Args:
        algorithm_configs_map: Dictionary mapping agent group names to their
            AlgorithmConfig instances. Group names must match the environment's
            group names exactly.

    Attributes:
        algorithm_configs_map: Dictionary storing algorithm configs per group.

    Note:
        All algorithms in the ensemble must share the same training paradigm
        (all on-policy or all off-policy). At least one action space type
        (continuous or discrete) must be supported by all algorithms.

    Example:
        Creating an ensemble config with MASAC for one group and MAPPO for another::

            from benchmarl.algorithms import MasacConfig, MappoConfig

            config = EnsembleAlgorithmConfig(
                algorithm_configs_map={
                    "predators": MasacConfig(
                        share_param_critic=False,
                        num_qvalue_nets=2,
                        loss_function="l2",
                    ),
                    "prey": MappoConfig(
                        share_param_critic=True,
                        clip_epsilon=0.2,
                    ),
                }
            )
    """

    algorithm_configs_map: Dict[str, AlgorithmConfig]

    def __post_init__(self):
        """Validate ensemble configuration after dataclass initialization.

        Ensures all algorithms in the ensemble are either on-policy or off-policy
        (not mixed), and that at least one algorithm supports the action space type.
        """
        algorithm_configs = list(self.algorithm_configs_map.values())
        self._on_policy = algorithm_configs[0].on_policy()

        for algorithm_config in algorithm_configs[1:]:
            if algorithm_config.on_policy() != self._on_policy:
                raise ValueError(
                    "Algorithms in EnsembleAlgorithmConfig must either be all on_policy or all off_policy"
                )

        if (
            not self.supports_discrete_actions()
            and not self.supports_continuous_actions()
        ):
            raise ValueError(
                "Ensemble algorithm does not support discrete actions nor continuous actions."
                " Make sure that at least one type of action is supported across all the algorithms used."
            )

    def get_algorithm(self, experiment) -> Algorithm:
        """Create and return the ensemble algorithm instance.

        Args:
            experiment: The experiment containing environment and group information.

        Returns:
            Configured EnsembleAlgorithm instance with algorithms for each group.

        Raises:
            ValueError: If algorithm config group names don't match environment group names.
        """
        if set(self.algorithm_configs_map.keys()) != set(experiment.group_map.keys()):
            raise ValueError(
                f"EnsembleAlgorithm group names {self.algorithm_configs_map.keys()} do not match "
                f"environment group names {experiment.group_map.keys()}"
            )
        return self.associated_class()(  # type: ignore[call-arg]  # EnsembleAlgorithm accepts algorithms_map
            algorithms_map={
                group: algorithm_config.get_algorithm(experiment)
                for group, algorithm_config in self.algorithm_configs_map.items()
            },
            experiment=experiment,
        )

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        """Load algorithm configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Raises:
            NotImplementedError: This method is not implemented for EnsembleAlgorithm.
        """
        raise NotImplementedError

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        """Return the algorithm class associated with this config.

        Returns:
            The EnsembleAlgorithm class.
        """
        return EnsembleAlgorithm

    @staticmethod
    def on_policy() -> bool:
        """Check if the ensemble algorithm is on-policy.

        Returns:
            True if all constituent algorithms are on-policy, False otherwise.

        Note: For ensemble algorithms, this should be determined from algorithm_configs_map.
              This method cannot be static in EnsembleAlgorithm but must be for API compatibility.
        """
        raise NotImplementedError(
            "EnsembleAlgorithm.on_policy() should not be called directly"
        )

    @staticmethod
    def supports_continuous_actions() -> bool:
        """Check if all algorithms in the ensemble support continuous actions.

        Returns:
            True if all constituent algorithms support continuous actions, False otherwise.

        Note: For ensemble algorithms, this should be determined from algorithm_configs_map.
              This method cannot be static in EnsembleAlgorithm but must be for API compatibility.
        """
        raise NotImplementedError(
            "EnsembleAlgorithm.supports_continuous_actions() should not be called directly"
        )

    @staticmethod
    def supports_discrete_actions() -> bool:
        """Check if all algorithms in the ensemble support discrete actions.

        Returns:
            True if all constituent algorithms support discrete actions, False otherwise.
        """
        raise NotImplementedError(
            "EnsembleAlgorithm.supports_discrete_actions() should not be called directly"
        )

    @staticmethod
    def has_independent_critic() -> bool:
        """Check if any algorithm in the ensemble has an independent critic.

        Returns:
            True if at least one constituent algorithm has an independent critic, False otherwise.
        """
        raise NotImplementedError(
            "EnsembleAlgorithm.has_independent_critic() should not be called directly"
        )

    @staticmethod
    def has_centralized_critic() -> bool:
        """Check if any algorithm in the ensemble has a centralized critic.

        Returns:
            True if at least one constituent algorithm has a centralized critic, False otherwise.
        """
        raise NotImplementedError(
            "EnsembleAlgorithm.has_centralized_critic() should not be called directly"
        )

    def has_critic(self) -> bool:
        """Check if any algorithm in the ensemble has a critic.

        Returns:
            True if at least one algorithm has a centralized or independent critic, False otherwise.
        """
        return self.has_centralized_critic() or self.has_independent_critic()
