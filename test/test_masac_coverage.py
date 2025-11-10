#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
Targeted tests to improve MASAC algorithm coverage.
Focuses on uncovered code paths including:
- Discrete action handling with coupled/decoupled values
- Action masking
- State-based critic configurations
- Edge cases in parameter handling
"""

import pytest
from torch import nn
from utils import _has_smacv2, _has_vmas

from benchmarl.algorithms import MasacConfig
from benchmarl.environments import Smacv2Task, VmasTask
from benchmarl.experiment import Experiment
from benchmarl.models import MlpConfig


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
class TestMasacCoverage:
    """Tests to achieve 80%+ coverage for MASAC algorithm."""

    def test_masac_discrete_coupled_values_share_critic_false_warning(
        self, experiment_config, mlp_sequence_config, capsys
    ):
        """Test warning when coupled_discrete_values=True and share_param_critic=False."""
        task = VmasTask.BALANCE.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = True
        algo_config.share_param_critic = False  # This should trigger warning

        with pytest.warns(
            UserWarning, match="disabling share_param_critic.*has not effect"
        ):
            experiment = Experiment(
                algorithm_config=algo_config,
                model_config=mlp_sequence_config,
                seed=0,
                config=experiment_config,
                task=task,
            )
            experiment.run()

    def test_masac_discrete_decoupled_values(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with discrete actions and decoupled value functions."""
        task = VmasTask.BALANCE.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = False  # Use decoupled value module
        algo_config.share_param_critic = True

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_discrete_decoupled_no_share_critic(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with discrete actions, decoupled values, no shared critic."""
        task = VmasTask.BALANCE.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = False
        algo_config.share_param_critic = False  # Don't share critic params

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_discrete_coupled_values(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with discrete actions and coupled value functions."""
        task = VmasTask.BALANCE.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = True  # Use coupled value module
        algo_config.share_param_critic = True

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_continuous_no_tanh_normal(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with continuous actions using IndependentNormal instead of TanhNormal."""
        task = VmasTask.BALANCE.get_from_yaml()
        experiment_config.prefer_continuous_actions = True

        algo_config = MasacConfig.get_from_yaml()
        algo_config.use_tanh_normal = False  # Use IndependentNormal distribution

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_continuous_share_critic_false(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with continuous actions and no parameter sharing in critic."""
        task = VmasTask.BALANCE.get_from_yaml()
        experiment_config.prefer_continuous_actions = True

        algo_config = MasacConfig.get_from_yaml()
        algo_config.share_param_critic = False  # Different critic output spec

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_fixed_alpha(self, experiment_config, mlp_sequence_config):
        """Test MASAC with fixed alpha (no alpha optimization)."""
        task = VmasTask.BALANCE.get_from_yaml()

        algo_config = MasacConfig.get_from_yaml()
        algo_config.fixed_alpha = True  # Don't optimize alpha
        algo_config.alpha_init = 0.2

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_custom_alpha_bounds(self, experiment_config, mlp_sequence_config):
        """Test MASAC with custom alpha min/max bounds."""
        task = VmasTask.BALANCE.get_from_yaml()

        algo_config = MasacConfig.get_from_yaml()
        algo_config.fixed_alpha = False
        algo_config.min_alpha = 0.01
        algo_config.max_alpha = 1.0

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_different_scale_mappings(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with different scale mapping functions."""
        task = VmasTask.BALANCE.get_from_yaml()
        experiment_config.prefer_continuous_actions = True

        for scale_mapping in ["exp", "softplus", "relu"]:
            algo_config = MasacConfig.get_from_yaml()
            algo_config.scale_mapping = scale_mapping

            experiment = Experiment(
                algorithm_config=algo_config,
                model_config=mlp_sequence_config,
                seed=0,
                config=experiment_config,
                task=task,
            )
            experiment.run()

    def test_masac_custom_loss_function(self, experiment_config, mlp_sequence_config):
        """Test MASAC with different loss functions."""
        task = VmasTask.BALANCE.get_from_yaml()

        for loss_fn in ["l2", "smooth_l1"]:
            algo_config = MasacConfig.get_from_yaml()
            algo_config.loss_function = loss_fn

            experiment = Experiment(
                algorithm_config=algo_config,
                model_config=mlp_sequence_config,
                seed=0,
                config=experiment_config,
                task=task,
            )
            experiment.run()

    # Note: delay_qvalue=False is not supported in current implementation
    # as it requires target networks. Skipping this test.

    def test_masac_custom_target_entropy(self, experiment_config, mlp_sequence_config):
        """Test MASAC with custom target entropy value."""
        task = VmasTask.BALANCE.get_from_yaml()

        algo_config = MasacConfig.get_from_yaml()
        algo_config.target_entropy = -2.0  # Custom target entropy (not "auto")

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_discrete_custom_target_entropy_weight(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC discrete with custom target entropy weight."""
        task = VmasTask.BALANCE.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.discrete_target_entropy_weight = 0.5  # Different from default 0.2

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_multiple_qvalue_nets(self, experiment_config, mlp_sequence_config):
        """Test MASAC with different numbers of Q-value networks."""
        task = VmasTask.BALANCE.get_from_yaml()

        for num_nets in [1, 3]:
            algo_config = MasacConfig.get_from_yaml()
            algo_config.num_qvalue_nets = num_nets

            experiment = Experiment(
                algorithm_config=algo_config,
                model_config=mlp_sequence_config,
                seed=0,
                config=experiment_config,
                task=task,
            )
            experiment.run()

    @pytest.mark.parametrize("task_name", [VmasTask.NAVIGATION, VmasTask.TRANSPORT])
    def test_masac_discrete_multiagent_tasks(
        self, task_name, experiment_config, mlp_sequence_config
    ):
        """Test MASAC on various multi-agent tasks with discrete actions."""
        task = task_name.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = True

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("task_name", [VmasTask.NAVIGATION, VmasTask.TRANSPORT])
    def test_masac_discrete_decoupled_multiagent(
        self, task_name, experiment_config, mlp_sequence_config
    ):
        """Test MASAC decoupled on multi-agent tasks."""
        task = task_name.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = False  # Test decoupled path

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_config_methods(self):
        """Test MasacConfig static methods for coverage."""
        assert MasacConfig.supports_continuous_actions() is True
        assert MasacConfig.supports_discrete_actions() is True
        assert MasacConfig.on_policy() is False
        assert MasacConfig.has_centralized_critic() is True

        from benchmarl.algorithms import Masac

        assert MasacConfig.associated_class() == Masac

    def test_masac_with_state_spec_continuous(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with state-based critic for continuous actions."""
        task = VmasTask.NAVIGATION.get_from_yaml()
        experiment_config.prefer_continuous_actions = True

        algo_config = MasacConfig.get_from_yaml()
        algo_config.share_param_critic = True

        # Use a separate critic model to test state-based paths
        critic_model_config = MlpConfig(
            num_cells=[16, 8], activation_class=nn.Tanh, layer_class=nn.Linear
        )

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_with_state_spec_discrete_coupled(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with state-based critic for discrete actions (coupled)."""
        task = VmasTask.NAVIGATION.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = True

        critic_model_config = MlpConfig(
            num_cells=[16, 8], activation_class=nn.Tanh, layer_class=nn.Linear
        )

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_with_state_spec_discrete_decoupled(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with state-based critic for discrete actions (decoupled)."""
        task = VmasTask.NAVIGATION.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = False

        critic_model_config = MlpConfig(
            num_cells=[16, 8], activation_class=nn.Tanh, layer_class=nn.Linear
        )

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()


@pytest.mark.skipif(not _has_smacv2, reason="SMACv2 not found")
class TestMasacCoverageSMACv2:
    """Additional MASAC tests using SMACv2 for state_spec and action_mask coverage."""

    def test_masac_smacv2_discrete_with_action_mask(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with discrete actions and action masking (SMACv2)."""
        task = Smacv2Task.TERRAN_5_VS_5.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = True

        critic_model_config = MlpConfig(
            num_cells=[32, 16], activation_class=nn.Tanh, layer_class=nn.Linear
        )

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_smacv2_discrete_coupled_with_state(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with state-based critic and coupled values (SMACv2)."""
        task = Smacv2Task.TERRAN_5_VS_5.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = True
        algo_config.share_param_critic = True

        critic_model_config = MlpConfig(
            num_cells=[32, 16], activation_class=nn.Tanh, layer_class=nn.Linear
        )

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_masac_smacv2_discrete_decoupled_with_state(
        self, experiment_config, mlp_sequence_config
    ):
        """Test MASAC with state-based critic and decoupled values (SMACv2)."""
        task = Smacv2Task.TERRAN_5_VS_5.get_from_yaml()
        experiment_config.prefer_continuous_actions = False

        algo_config = MasacConfig.get_from_yaml()
        algo_config.coupled_discrete_values = False  # Decoupled with state
        algo_config.share_param_critic = True

        critic_model_config = MlpConfig(
            num_cells=[32, 16], activation_class=nn.Tanh, layer_class=nn.Linear
        )

        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
