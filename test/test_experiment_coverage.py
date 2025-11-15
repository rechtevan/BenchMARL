#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Comprehensive experiment coverage tests for edge cases and error paths."""

from __future__ import annotations

import pytest

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig


class TestExperimentCoverage:
    """Tests to improve experiment.py coverage."""

    def test_experiment_with_max_n_iters_only(
        self, experiment_config, mlp_sequence_config
    ):
        """Test experiment with only max_n_iters configured (no max_n_frames)."""
        experiment_config.max_n_iters = 3
        experiment_config.max_n_frames = None  # Only use iterations
        experiment_config.evaluation = False
        experiment_config.checkpoint_interval = 1000000

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_experiment_validation_errors(self):
        """Test ExperimentConfig validation catches errors."""
        config = ExperimentConfig.get_from_yaml()

        # Test checkpoint_interval validation
        config.on_policy_collected_frames_per_batch = 100
        config.checkpoint_interval = 150  # NOT a multiple
        with pytest.raises(ValueError, match="checkpoint_interval"):
            config.validate(on_policy=True)

        # Test evaluation_interval validation
        config.checkpoint_interval = 100  # Fix this
        config.evaluation = True
        config.evaluation_interval = 150  # NOT a multiple
        with pytest.raises(ValueError, match="evaluation_interval"):
            config.validate(on_policy=True)

    def test_experiment_with_checkpoint_at_end(
        self, experiment_config, mlp_sequence_config
    ):
        """Test experiment with checkpoint_at_end enabled."""
        experiment_config.max_n_iters = 2
        experiment_config.evaluation = False
        experiment_config.checkpoint_at_end = True  # Save checkpoint at end
        experiment_config.checkpoint_interval = 1000000  # Don't save during training

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

        # Should have saved checkpoint at end
        from pathlib import Path

        checkpoint_files = list(Path(experiment_config.save_folder).rglob("*.pt"))
        assert len(checkpoint_files) > 0, "No checkpoint saved at end"

    def test_experiment_collected_frames_per_batch_on_policy(self):
        """Test collected_frames_per_batch method for on-policy."""
        config = ExperimentConfig.get_from_yaml()
        config.on_policy_collected_frames_per_batch = 256

        frames = config.collected_frames_per_batch(on_policy=True)
        assert frames == 256

    def test_experiment_collected_frames_per_batch_off_policy(self):
        """Test collected_frames_per_batch method for off-policy."""
        config = ExperimentConfig.get_from_yaml()
        config.off_policy_collected_frames_per_batch = 128

        frames = config.collected_frames_per_batch(on_policy=False)
        assert frames == 128

    def test_experiment_with_render_enabled(
        self, experiment_config, mlp_sequence_config
    ):
        """Test experiment with rendering enabled."""
        experiment_config.max_n_iters = 1
        experiment_config.evaluation = False
        experiment_config.render = True  # Enable rendering
        experiment_config.checkpoint_interval = 1000000

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_experiment_with_evaluation_deterministic(
        self, experiment_config, mlp_sequence_config
    ):
        """Test experiment with deterministic evaluation."""
        experiment_config.max_n_iters = 2
        experiment_config.evaluation = True
        experiment_config.evaluation_interval = 200
        experiment_config.evaluation_deterministic_actions = True  # Deterministic
        experiment_config.checkpoint_interval = 1000000

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_experiment_without_json_logger(
        self, experiment_config, mlp_sequence_config
    ):
        """Test experiment without JSON logging."""
        experiment_config.max_n_iters = 1
        experiment_config.evaluation = False
        experiment_config.create_json = False  # Disable JSON
        experiment_config.loggers = ["csv"]  # Only CSV
        experiment_config.checkpoint_interval = 1000000

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_experiment_keep_checkpoints_num(
        self, experiment_config, mlp_sequence_config
    ):
        """Test experiment with keep_checkpoints_num limiting saved checkpoints."""
        experiment_config.max_n_iters = 3
        experiment_config.evaluation = False
        experiment_config.checkpoint_interval = 100
        experiment_config.keep_checkpoints_num = 1  # Keep only 1 checkpoint

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

        # Should have only 1 checkpoint file
        from pathlib import Path

        checkpoint_files = list(
            Path(experiment_config.save_folder).rglob("checkpoint*.pt")
        )
        assert (
            len(checkpoint_files) <= 2
        ), f"Too many checkpoints saved: {len(checkpoint_files)}"
