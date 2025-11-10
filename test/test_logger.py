#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict
from torchrl.record import TensorboardLogger
from torchrl.record.loggers.wandb import WandbLogger

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.logger import JsonWriter, Logger
from benchmarl.models import MlpConfig


class TestLogger:
    """Test suite for Logger class to improve coverage from 79.17% to 80%+"""

    def test_logger_initialization_empty_loggers(self, tmp_path):
        """Test logger initialization with no loggers configured (line 62)"""
        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.loggers = []  # Empty loggers list
        experiment_config.create_json = False

        logger = Logger(
            experiment_name="test_exp",
            folder_name=str(tmp_path),
            experiment_config=experiment_config,
            algorithm_name="test_algo",
            environment_name="test_env",
            task_name="test_task",
            model_name="test_model",
            group_map={"group1": ["agent1"]},
            seed=0,
            project_name="test_project",
            wandb_extra_kwargs={},
        )

        assert len(logger.loggers) == 0
        assert logger.json_writer is None

    def test_logger_wandb_project_validation(self, tmp_path):
        """Test ValueError when wandb_extra_kwargs.project differs from project_name (lines 64-67)"""
        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.loggers = ["wandb"]
        experiment_config.create_json = False

        with pytest.raises(
            ValueError,
            match="wandb_extra_kwargs.project.*is different from the project_name",
        ):
            Logger(
                experiment_name="test_exp",
                folder_name=str(tmp_path),
                experiment_config=experiment_config,
                algorithm_name="test_algo",
                environment_name="test_env",
                task_name="test_task",
                model_name="test_model",
                group_map={"group1": ["agent1"]},
                seed=0,
                project_name="test_project",
                wandb_extra_kwargs={"project": "different_project"},
            )

    def test_log_hparams_tensorboard_flattening(self, tmp_path):
        """Test log_hparams with TensorboardLogger - flattening dicts and sequences (lines 89-109)"""
        # Create a mock TensorboardLogger
        mock_tb_logger = MagicMock(spec=TensorboardLogger)

        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.loggers = []  # Don't use get_logger
        experiment_config.create_json = False

        logger = Logger(
            experiment_name="test_exp",
            folder_name=str(tmp_path),
            experiment_config=experiment_config,
            algorithm_name="test_algo",
            environment_name="test_env",
            task_name="test_task",
            model_name="test_model",
            group_map={"group1": ["agent1"]},
            seed=0,
            project_name="test_project",
            wandb_extra_kwargs={},
        )

        # Manually add the mock tensorboard logger
        logger.loggers = [mock_tb_logger]

        # Test with different data types to exercise the flattening logic
        # Lines 102-104: Non-(int, float, str, Tensor) values get converted to str
        # Then flatten function (lines 86-99) processes them:
        # - MutableMapping types get recursively flattened
        # - Sequence types (including strings!) get indexed
        # - Other types pass through
        hparams = {
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "hello",  # String is a Sequence, will become str_val_0, str_val_1, etc.
            "tensor_val": torch.tensor([1.0]),  # Tensor passes through line 103 check
            "dict_val": {
                "a": 1
            },  # Will be converted to str (line 104), then treated as Sequence in flatten
            "list_val": [
                1,
                2,
            ],  # Will be converted to str (line 104), then treated as Sequence in flatten
        }

        logger.log_hparams(**hparams)

        # Verify log_hparams was called
        assert mock_tb_logger.log_hparams.called
        # Get the flattened dict that was passed
        call_args = mock_tb_logger.log_hparams.call_args[0][0]

        # Verify the conversion and flattening happened:
        # - int and float pass through unchanged
        assert "int_val" in call_args
        assert call_args["int_val"] == 42
        assert "float_val" in call_args
        assert call_args["float_val"] == 3.14

        # - Tensor passes through (line 103)
        assert "tensor_val" in call_args

        # - String, dict, and list get treated as Sequences and indexed
        # This exercises lines 94-96 (Sequence handling)
        assert any(key.startswith("str_val_") for key in call_args.keys())
        assert any(key.startswith("dict_val_") for key in call_args.keys())
        assert any(key.startswith("list_val_") for key in call_args.keys())

    def test_log_training_empty_loggers(self, tmp_path):
        """Test log_training with no loggers returns early (line 165-166)"""
        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.loggers = []
        experiment_config.create_json = False

        logger = Logger(
            experiment_name="test_exp",
            folder_name=str(tmp_path),
            experiment_config=experiment_config,
            algorithm_name="test_algo",
            environment_name="test_env",
            task_name="test_task",
            model_name="test_model",
            group_map={"group1": ["agent1"]},
            seed=0,
            project_name="test_project",
            wandb_extra_kwargs={},
        )

        # Create a simple training_td
        training_td = TensorDict(
            {"loss": torch.tensor([1.0, 2.0])},
            batch_size=[2],
        )

        # Should return early without error
        logger.log_training(group="group1", training_td=training_td, step=1)

    def test_log_training_with_items(self, tmp_path):
        """Test log_training with training_td items (line 169)"""
        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.loggers = []
        experiment_config.create_json = False

        mock_csv_logger = MagicMock()

        logger = Logger(
            experiment_name="test_exp",
            folder_name=str(tmp_path),
            experiment_config=experiment_config,
            algorithm_name="test_algo",
            environment_name="test_env",
            task_name="test_task",
            model_name="test_model",
            group_map={"group1": ["agent1"]},
            seed=0,
            project_name="test_project",
            wandb_extra_kwargs={},
        )

        # Manually add the mock csv logger
        logger.loggers = [mock_csv_logger]

        training_td = TensorDict(
            {
                "loss": torch.tensor([1.0, 2.0]),
                "grad_norm": torch.tensor([0.5, 0.6]),
            },
            batch_size=[2],
        )

        logger.log_training(group="group1", training_td=training_td, step=1)

        # Verify log_scalar was called for each metric
        assert mock_csv_logger.log_scalar.call_count >= 2

    def test_log_evaluation_empty_rollouts(self, tmp_path):
        """Test log_evaluation with empty rollouts returns early (line 180-183)"""
        mock_logger = MagicMock()

        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.loggers = []
        experiment_config.create_json = False

        logger = Logger(
            experiment_name="test_exp",
            folder_name=str(tmp_path),
            experiment_config=experiment_config,
            algorithm_name="test_algo",
            environment_name="test_env",
            task_name="test_task",
            model_name="test_model",
            group_map={"group1": ["agent1"]},
            seed=0,
            project_name="test_project",
            wandb_extra_kwargs={},
        )

        # Manually add the mock logger
        logger.loggers = [mock_logger]

        # Empty rollouts list
        logger.log_evaluation(rollouts=[], total_frames=100, step=1, video_frames=None)

        # Should not call log methods since rollouts is empty
        assert mock_logger.log_scalar.call_count == 0

    def test_commit_with_wandb(self, tmp_path):
        """Test commit() with WandbLogger (lines 258-261, 264)"""
        mock_wandb_logger = MagicMock(spec=WandbLogger)
        mock_wandb_logger.experiment = MagicMock()

        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.loggers = []
        experiment_config.create_json = False

        logger = Logger(
            experiment_name="test_exp",
            folder_name=str(tmp_path),
            experiment_config=experiment_config,
            algorithm_name="test_algo",
            environment_name="test_env",
            task_name="test_task",
            model_name="test_model",
            group_map={"group1": ["agent1"]},
            seed=0,
            project_name="test_project",
            wandb_extra_kwargs={},
        )

        # Manually add the mock wandb logger
        logger.loggers = [mock_wandb_logger]

        logger.commit()

        # Verify wandb commit was called
        mock_wandb_logger.experiment.log.assert_called_once_with({}, commit=True)

    def test_log_with_wandb(self, tmp_path):
        """Test log() with WandbLogger (lines 263-266, 269)"""
        mock_wandb_logger = MagicMock(spec=WandbLogger)
        mock_wandb_logger.experiment = MagicMock()

        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.loggers = []
        experiment_config.create_json = False

        logger = Logger(
            experiment_name="test_exp",
            folder_name=str(tmp_path),
            experiment_config=experiment_config,
            algorithm_name="test_algo",
            environment_name="test_env",
            task_name="test_task",
            model_name="test_model",
            group_map={"group1": ["agent1"]},
            seed=0,
            project_name="test_project",
            wandb_extra_kwargs={},
        )

        # Manually add the mock wandb logger
        logger.loggers = [mock_wandb_logger]

        test_dict = {"metric1": 1.0, "metric2": 2.0}
        logger.log(test_dict, step=1)

        # Verify wandb log was called with commit=False
        mock_wandb_logger.experiment.log.assert_called_once_with(
            test_dict, commit=False
        )

    def test_finish_with_wandb(self, tmp_path):
        """Test finish() with WandbLogger (lines 271-276, 277)"""
        with patch("wandb.finish") as mock_wandb_finish:
            mock_wandb_logger = MagicMock(spec=WandbLogger)

            experiment_config = ExperimentConfig.get_from_yaml()
            experiment_config.loggers = []
            experiment_config.create_json = False

            logger = Logger(
                experiment_name="test_exp",
                folder_name=str(tmp_path),
                experiment_config=experiment_config,
                algorithm_name="test_algo",
                environment_name="test_env",
                task_name="test_task",
                model_name="test_model",
                group_map={"group1": ["agent1"]},
                seed=0,
                project_name="test_project",
                wandb_extra_kwargs={},
            )

            # Manually add the mock wandb logger
            logger.loggers = [mock_wandb_logger]

            logger.finish()

            # Verify wandb.finish was called
            mock_wandb_finish.assert_called_once()

    def test_log_hparams_non_tensorboard(self, tmp_path):
        """Test log_hparams with non-TensorboardLogger (line 108)"""
        mock_csv_logger = MagicMock()

        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.loggers = []
        experiment_config.create_json = False

        logger = Logger(
            experiment_name="test_exp",
            folder_name=str(tmp_path),
            experiment_config=experiment_config,
            algorithm_name="test_algo",
            environment_name="test_env",
            task_name="test_task",
            model_name="test_model",
            group_map={"group1": ["agent1"]},
            seed=0,
            project_name="test_project",
            wandb_extra_kwargs={},
        )

        # Manually add non-tensorboard logger
        logger.loggers = [mock_csv_logger]

        hparams = {"key1": "value1", "key2": 42}
        logger.log_hparams(**hparams)

        # Verify log_hparams was called with the dict as-is (no flattening)
        mock_csv_logger.log_hparams.assert_called_once()
        call_args = mock_csv_logger.log_hparams.call_args[0][0]
        assert call_args == hparams


class TestLoggerIntegration:
    """Integration tests for logger with actual experiments"""

    def test_experiment_with_empty_loggers(self, tmp_path):
        """Test logger with no loggers (covers line 62)"""
        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.save_folder = str(tmp_path)
        experiment_config.max_n_iters = 1
        experiment_config.loggers = []  # Empty loggers list
        experiment_config.create_json = False
        experiment_config.on_policy_collected_frames_per_batch = 50
        experiment_config.on_policy_n_envs_per_worker = 2
        experiment_config.on_policy_n_minibatch_iters = 1
        experiment_config.on_policy_minibatch_size = 2
        experiment_config.evaluation = False
        experiment_config.parallel_collection = False

        task = VmasTask.BALANCE.get_from_yaml()
        model_config = MlpConfig(
            num_cells=[8], layer_class=torch.nn.Linear, activation_class=torch.nn.Tanh
        )

        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

        # Verify no loggers were created
        assert len(experiment.logger.loggers) == 0

    def test_experiment_with_tensorboard_logger(self, tmp_path):
        """Test logger with tensorboard through actual experiment (covers lines 86-106)"""
        pytest.importorskip("tensorboard")  # Skip if tensorboard not available

        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.save_folder = str(tmp_path)
        experiment_config.max_n_iters = 1
        experiment_config.loggers = ["tensorboard"]
        experiment_config.create_json = False
        experiment_config.on_policy_collected_frames_per_batch = 50
        experiment_config.on_policy_n_envs_per_worker = 2
        experiment_config.on_policy_n_minibatch_iters = 1
        experiment_config.on_policy_minibatch_size = 2
        experiment_config.evaluation = False
        experiment_config.parallel_collection = False

        task = VmasTask.BALANCE.get_from_yaml()
        model_config = MlpConfig(
            num_cells=[8], layer_class=torch.nn.Linear, activation_class=torch.nn.Tanh
        )

        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

        # Verify tensorboard logger was used
        assert len(experiment.logger.loggers) > 0


class TestJsonWriter:
    """Test suite for JsonWriter class"""

    def test_json_writer_basic(self, tmp_path):
        """Test basic JsonWriter functionality"""
        writer = JsonWriter(
            folder=str(tmp_path),
            name="test.json",
            algorithm_name="algo1",
            task_name="task1",
            environment_name="env1",
            seed=42,
        )

        assert writer.path == tmp_path / "test.json"
        assert "env1" in writer.data
        assert "task1" in writer.data["env1"]
        assert "algo1" in writer.data["env1"]["task1"]
        assert "seed_42" in writer.data["env1"]["task1"]["algo1"]

    def test_json_writer_write_and_update(self, tmp_path):
        """Test JsonWriter write and update logic (lines 421-453)"""
        writer = JsonWriter(
            folder=str(tmp_path),
            name="test.json",
            algorithm_name="algo1",
            task_name="task1",
            environment_name="env1",
            seed=42,
        )

        # First write
        metrics = {
            "return": torch.tensor([1.0, 2.0, 3.0]),
            "group1_return": torch.tensor([1.5, 2.5, 3.5]),
        }
        writer.write(total_frames=100, metrics=metrics, evaluation_step=0)

        # Verify file exists and has correct content
        assert writer.path.exists()
        with open(writer.path) as f:
            data = json.load(f)

        assert "step_0" in data["env1"]["task1"]["algo1"]["seed_42"]
        step_data = data["env1"]["task1"]["algo1"]["seed_42"]["step_0"]
        assert step_data["step_count"] == 100
        assert step_data["return"] == [1.0, 2.0, 3.0]
        assert step_data["group1_return"] == [1.5, 2.5, 3.5]

        # Check absolute_metrics
        abs_metrics = data["env1"]["task1"]["algo1"]["seed_42"]["absolute_metrics"]
        assert abs_metrics["return"] == [3.0]
        assert abs_metrics["group1_return"] == [3.5]

        # Second write with higher values
        metrics2 = {
            "return": torch.tensor([2.0, 3.0, 4.0]),
            "group1_return": torch.tensor([2.5, 3.5, 4.5]),
        }
        writer.write(total_frames=200, metrics=metrics2, evaluation_step=1)

        with open(writer.path) as f:
            data = json.load(f)

        # Check updated absolute_metrics (should keep max)
        abs_metrics = data["env1"]["task1"]["algo1"]["seed_42"]["absolute_metrics"]
        assert abs_metrics["return"] == [4.0]
        assert abs_metrics["group1_return"] == [4.5]

    def test_json_writer_empty_metrics(self, tmp_path):
        """Test JsonWriter with empty metrics list (line 443-444)"""
        writer = JsonWriter(
            folder=str(tmp_path),
            name="test.json",
            algorithm_name="algo1",
            task_name="task1",
            environment_name="env1",
            seed=42,
        )

        # Write with empty metrics
        metrics = {
            "return": torch.tensor([]),
            "other_metric": torch.tensor([1.0]),
        }
        writer.write(total_frames=100, metrics=metrics, evaluation_step=0)

        with open(writer.path) as f:
            data = json.load(f)

        # Empty metric should not update absolute_metrics
        abs_metrics = data["env1"]["task1"]["algo1"]["seed_42"]["absolute_metrics"]
        assert "return" not in abs_metrics or abs_metrics.get("return") == []
        assert abs_metrics["other_metric"] == [1.0]

    def test_json_writer_update_existing_step(self, tmp_path):
        """Test JsonWriter updating existing step (lines 437-440)"""
        writer = JsonWriter(
            folder=str(tmp_path),
            name="test.json",
            algorithm_name="algo1",
            task_name="task1",
            environment_name="env1",
            seed=42,
        )

        # First write
        metrics1 = {"metric1": torch.tensor([1.0])}
        writer.write(total_frames=100, metrics=metrics1, evaluation_step=0)

        # Update same step with additional metric
        metrics2 = {"metric2": torch.tensor([2.0])}
        writer.write(total_frames=100, metrics=metrics2, evaluation_step=0)

        with open(writer.path) as f:
            data = json.load(f)

        step_data = data["env1"]["task1"]["algo1"]["seed_42"]["step_0"]
        # Should have both metrics
        assert "metric1" in step_data
        assert "metric2" in step_data
