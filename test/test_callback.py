#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest
import torch
from tensordict import TensorDict
from utils import _has_vmas

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback, CallbackNotifier


class TestCallback:
    """Test suite for Callback class."""

    def test_callback_initialization(self):
        """Test that Callback initializes with experiment set to None."""
        callback = Callback()
        assert callback.experiment is None

    def test_callback_on_setup(self):
        """Test that on_setup can be called without error."""
        callback = Callback()
        # Should not raise any exception
        callback.on_setup()

    def test_callback_on_batch_collected(self):
        """Test that on_batch_collected can be called without error."""
        callback = Callback()
        batch = TensorDict({"data": torch.tensor([1, 2, 3])}, batch_size=[3])
        # Should not raise any exception
        callback.on_batch_collected(batch)

    def test_callback_on_train_step(self):
        """Test that on_train_step can be called and returns None."""
        callback = Callback()
        batch = TensorDict({"data": torch.tensor([1, 2, 3])}, batch_size=[3])
        result = callback.on_train_step(batch, "test_group")
        # Base implementation returns None (pass statement)
        assert result is None

    def test_callback_on_train_end(self):
        """Test that on_train_end can be called without error."""
        callback = Callback()
        training_td = TensorDict({"loss": torch.tensor(0.5)}, batch_size=[])
        # Should not raise any exception
        callback.on_train_end(training_td, "test_group")

    def test_callback_on_evaluation_end(self):
        """Test that on_evaluation_end can be called without error."""
        callback = Callback()
        rollouts = [
            TensorDict({"reward": torch.tensor(1.0)}, batch_size=[]),
            TensorDict({"reward": torch.tensor(2.0)}, batch_size=[]),
        ]
        # Should not raise any exception
        callback.on_evaluation_end(rollouts)


class CustomCallbackA(Callback):
    """Custom callback for testing that tracks all lifecycle calls."""

    def __init__(self):
        super().__init__()
        self.setup_called = False
        self.batch_collected_count = 0
        self.train_step_count = 0
        self.train_end_count = 0
        self.evaluation_end_count = 0

    def on_setup(self):
        self.setup_called = True

    def on_batch_collected(self, batch):
        self.batch_collected_count += 1

    def on_train_step(self, batch, group):
        self.train_step_count += 1
        return TensorDict({"loss_a": torch.tensor(0.1)}, batch_size=[])

    def on_train_end(self, training_td, group):
        self.train_end_count += 1

    def on_evaluation_end(self, rollouts):
        self.evaluation_end_count += 1


class CustomCallbackB(Callback):
    """Another custom callback for testing multiple callbacks."""

    def __init__(self):
        super().__init__()
        self.setup_called = False
        self.train_step_count = 0

    def on_setup(self):
        self.setup_called = True

    def on_train_step(self, batch, group):
        self.train_step_count += 1
        return TensorDict({"loss_b": torch.tensor(0.2)}, batch_size=[])


class CustomCallbackNoReturn(Callback):
    """Callback that doesn't return anything from on_train_step."""

    def __init__(self):
        super().__init__()
        self.train_step_count = 0

    def on_train_step(self, batch, group):
        self.train_step_count += 1
        # Return None explicitly
        return None


class TestCallbackNotifier:
    """Test suite for CallbackNotifier class."""

    def test_callback_notifier_initialization(self):
        """Test that CallbackNotifier initializes callbacks correctly."""
        callback1 = CustomCallbackA()
        callback2 = CustomCallbackB()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback1, callback2])

        assert notifier.callbacks == [callback1, callback2]
        assert callback1.experiment == experiment
        assert callback2.experiment == experiment

    def test_callback_notifier_empty_list(self):
        """Test that CallbackNotifier works with empty callback list."""
        experiment = "mock_experiment"
        notifier = CallbackNotifier(experiment, [])
        assert notifier.callbacks == []

    def test_on_setup_notification(self):
        """Test that _on_setup calls all callbacks' on_setup methods."""
        callback1 = CustomCallbackA()
        callback2 = CustomCallbackB()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback1, callback2])
        notifier._on_setup()

        assert callback1.setup_called
        assert callback2.setup_called

    def test_on_batch_collected_notification(self):
        """Test that _on_batch_collected calls all callbacks."""
        callback1 = CustomCallbackA()
        callback2 = CustomCallbackA()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback1, callback2])

        batch = TensorDict({"data": torch.tensor([1, 2, 3])}, batch_size=[3])
        notifier._on_batch_collected(batch)

        assert callback1.batch_collected_count == 1
        assert callback2.batch_collected_count == 1

        # Call again to verify it accumulates
        notifier._on_batch_collected(batch)
        assert callback1.batch_collected_count == 2
        assert callback2.batch_collected_count == 2

    def test_on_train_step_single_callback_with_return(self):
        """Test _on_train_step with a single callback that returns data."""
        callback = CustomCallbackA()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback])

        batch = TensorDict({"data": torch.tensor([1, 2, 3])}, batch_size=[3])
        result = notifier._on_train_step(batch, "test_group")

        assert callback.train_step_count == 1
        assert result is not None
        assert "loss_a" in result.keys()
        assert torch.isclose(result["loss_a"], torch.tensor(0.1))

    def test_on_train_step_multiple_callbacks_merge(self):
        """Test _on_train_step with multiple callbacks merges TensorDicts."""
        callback1 = CustomCallbackA()
        callback2 = CustomCallbackB()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback1, callback2])

        batch = TensorDict({"data": torch.tensor([1, 2, 3])}, batch_size=[3])
        result = notifier._on_train_step(batch, "test_group")

        assert callback1.train_step_count == 1
        assert callback2.train_step_count == 1
        assert result is not None
        assert "loss_a" in result.keys()
        assert "loss_b" in result.keys()
        assert torch.isclose(result["loss_a"], torch.tensor(0.1))
        assert torch.isclose(result["loss_b"], torch.tensor(0.2))

    def test_on_train_step_callback_returns_none(self):
        """Test _on_train_step when callback returns None."""
        callback = CustomCallbackNoReturn()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback])

        batch = TensorDict({"data": torch.tensor([1, 2, 3])}, batch_size=[3])
        result = notifier._on_train_step(batch, "test_group")

        assert callback.train_step_count == 1
        assert result is None

    def test_on_train_step_mixed_returns(self):
        """Test _on_train_step with mix of None and TensorDict returns."""
        callback1 = CustomCallbackNoReturn()  # Returns None
        callback2 = CustomCallbackA()  # Returns TensorDict
        callback3 = CustomCallbackB()  # Returns TensorDict
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback1, callback2, callback3])

        batch = TensorDict({"data": torch.tensor([1, 2, 3])}, batch_size=[3])
        result = notifier._on_train_step(batch, "test_group")

        assert callback1.train_step_count == 1
        assert callback2.train_step_count == 1
        assert callback3.train_step_count == 1
        assert result is not None
        assert "loss_a" in result.keys()
        assert "loss_b" in result.keys()

    def test_on_train_step_update_existing_tensordict(self):
        """Test _on_train_step updates TensorDict when multiple callbacks return data."""

        class CallbackWithLoss1(Callback):
            def on_train_step(self, batch, group):
                return TensorDict({"loss1": torch.tensor(1.0)}, batch_size=[])

        class CallbackWithLoss2(Callback):
            def on_train_step(self, batch, group):
                return TensorDict({"loss2": torch.tensor(2.0)}, batch_size=[])

        class CallbackWithLoss3(Callback):
            def on_train_step(self, batch, group):
                return TensorDict({"loss3": torch.tensor(3.0)}, batch_size=[])

        callback1 = CallbackWithLoss1()
        callback2 = CallbackWithLoss2()
        callback3 = CallbackWithLoss3()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback1, callback2, callback3])

        batch = TensorDict({"data": torch.tensor([1, 2, 3])}, batch_size=[3])
        result = notifier._on_train_step(batch, "test_group")

        # All losses should be present in the merged result
        assert result is not None
        assert "loss1" in result.keys()
        assert "loss2" in result.keys()
        assert "loss3" in result.keys()
        assert result["loss1"].item() == 1.0
        assert result["loss2"].item() == 2.0
        assert result["loss3"].item() == 3.0

    def test_on_train_end_notification(self):
        """Test that _on_train_end calls all callbacks."""
        callback1 = CustomCallbackA()
        callback2 = CustomCallbackA()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback1, callback2])

        training_td = TensorDict({"loss": torch.tensor(0.5)}, batch_size=[])
        notifier._on_train_end(training_td, "test_group")

        assert callback1.train_end_count == 1
        assert callback2.train_end_count == 1

    def test_on_evaluation_end_notification(self):
        """Test that _on_evaluation_end calls all callbacks."""
        callback1 = CustomCallbackA()
        callback2 = CustomCallbackA()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback1, callback2])

        rollouts = [
            TensorDict({"reward": torch.tensor(1.0)}, batch_size=[]),
            TensorDict({"reward": torch.tensor(2.0)}, batch_size=[]),
        ]
        notifier._on_evaluation_end(rollouts)

        assert callback1.evaluation_end_count == 1
        assert callback2.evaluation_end_count == 1

    def test_callback_execution_order(self):
        """Test that callbacks are executed in the order they are registered."""
        execution_order = []

        class OrderTrackingCallback1(Callback):
            def on_setup(self):
                execution_order.append("callback1")

        class OrderTrackingCallback2(Callback):
            def on_setup(self):
                execution_order.append("callback2")

        class OrderTrackingCallback3(Callback):
            def on_setup(self):
                execution_order.append("callback3")

        callback1 = OrderTrackingCallback1()
        callback2 = OrderTrackingCallback2()
        callback3 = OrderTrackingCallback3()
        experiment = "mock_experiment"

        notifier = CallbackNotifier(experiment, [callback1, callback2, callback3])
        notifier._on_setup()

        assert execution_order == ["callback1", "callback2", "callback3"]


class ExperimentAccessCallback(Callback):
    """Callback for testing access to experiment object."""

    def __init__(self):
        super().__init__()
        self.has_policy = False
        self.has_collector = False

    def on_setup(self):
        # Verify we can access experiment attributes
        if hasattr(self.experiment, "policy") and self.experiment.policy is not None:
            self.has_policy = True
        if (
            hasattr(self.experiment, "collector")
            and self.experiment.collector is not None
        ):
            self.has_collector = True


class MinimalCallback(Callback):
    """Minimal callback that only overrides one method."""

    def __init__(self):
        super().__init__()
        self.batch_count = 0

    def on_batch_collected(self, batch):
        self.batch_count += 1
        # Other methods use default implementation


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
class TestCallbackIntegration:
    """Integration tests for callbacks with real experiments."""

    def test_callback_with_experiment(self, experiment_config, mlp_sequence_config):
        """Test that callbacks work correctly within a full experiment."""
        callback1 = CustomCallbackA()
        callback2 = CustomCallbackB()

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
            callbacks=[callback1, callback2],
        )
        experiment.run()

        # Verify that all lifecycle methods were called
        assert callback1.setup_called
        assert callback2.setup_called
        assert callback1.batch_collected_count > 0
        assert callback1.train_step_count > 0
        assert callback2.train_step_count > 0
        # Note: evaluation might not run depending on config
        # assert callback1.evaluation_end_count > 0

    def test_callback_access_to_experiment(
        self, experiment_config, mlp_sequence_config
    ):
        """Test that callbacks can access the experiment object."""
        callback = ExperimentAccessCallback()

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
            callbacks=[callback],
        )
        experiment.run()

        assert callback.has_policy
        assert callback.has_collector

    def test_multiple_callbacks_in_experiment(
        self, experiment_config, mlp_sequence_config
    ):
        """Test that multiple callbacks work together in an experiment."""
        callback1 = CustomCallbackA()
        callback2 = CustomCallbackB()
        callback3 = CustomCallbackNoReturn()

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
            callbacks=[callback1, callback2, callback3],
        )
        experiment.run()

        # All callbacks should have been invoked
        assert callback1.setup_called
        assert callback2.setup_called
        assert callback1.train_step_count > 0
        assert callback2.train_step_count > 0
        assert callback3.train_step_count > 0

    def test_experiment_without_callbacks(self, experiment_config, mlp_sequence_config):
        """Test that experiments work normally without any callbacks."""
        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
            callbacks=[],
        )
        # Should run without errors
        experiment.run()

    def test_callback_inheritance(self, experiment_config, mlp_sequence_config):
        """Test that inheriting from Callback and overriding only some methods works."""
        callback = MinimalCallback()

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
            callbacks=[callback],
        )
        experiment.run()

        assert callback.batch_count > 0
