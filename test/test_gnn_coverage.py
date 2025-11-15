#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Comprehensive GNN model coverage tests."""

from __future__ import annotations

import importlib

import pytest
from torch import nn

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.models import GnnConfig, MlpConfig
from benchmarl.models.common import SequenceModelConfig


_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None

if _has_torch_geometric:
    import torch_geometric.nn.conv


@pytest.mark.skipif(not _has_torch_geometric, reason="torch_geometric not installed")
class TestGnnCoverage:
    """Tests to improve GNN model coverage."""

    def test_gnn_empty_topology(self, experiment_config):
        """Test GNN with empty topology (no edges)."""
        if not _has_torch_geometric:
            pytest.skip("torch_geometric not installed")

        gnn_config = SequenceModelConfig(
            model_configs=[
                GnnConfig(
                    topology="empty",  # No edges
                    self_loops=True,
                    gnn_class=torch_geometric.nn.conv.GATv2Conv,
                ),
                MlpConfig(
                    num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear
                ),
            ],
            intermediate_sizes=[5],
        )

        experiment_config.max_n_iters = 1
        experiment_config.evaluation = False

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=MappoConfig.get_from_yaml(),
            model_config=gnn_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_gnn_no_self_loops(self, experiment_config):
        """Test GNN with self_loops disabled."""
        if not _has_torch_geometric:
            pytest.skip("torch_geometric not installed")

        gnn_config = SequenceModelConfig(
            model_configs=[
                GnnConfig(
                    topology="full",
                    self_loops=False,  # Test without self loops
                    gnn_class=torch_geometric.nn.conv.GATv2Conv,
                ),
                MlpConfig(
                    num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear
                ),
            ],
            intermediate_sizes=[5],
        )

        experiment_config.max_n_iters = 1
        experiment_config.evaluation = False

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=MappoConfig.get_from_yaml(),
            model_config=gnn_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_gnn_with_different_gnn_classes(self, experiment_config):
        """Test GNN with different convolution classes."""
        if not _has_torch_geometric:
            pytest.skip("torch_geometric not installed")

        # Test with GCNConv (different from default GATv2Conv)
        gnn_config = SequenceModelConfig(
            model_configs=[
                GnnConfig(
                    topology="full",
                    self_loops=True,
                    gnn_class=torch_geometric.nn.conv.GCNConv,
                ),
                MlpConfig(
                    num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear
                ),
            ],
            intermediate_sizes=[5],
        )

        experiment_config.max_n_iters = 1
        experiment_config.evaluation = False

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=MappoConfig.get_from_yaml(),
            model_config=gnn_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_gnn_with_gnn_kwargs(self, experiment_config):
        """Test GNN with custom gnn_kwargs passed to convolution."""
        if not _has_torch_geometric:
            pytest.skip("torch_geometric not installed")

        gnn_config = SequenceModelConfig(
            model_configs=[
                GnnConfig(
                    topology="full",
                    self_loops=True,
                    gnn_class=torch_geometric.nn.conv.GATv2Conv,
                    gnn_kwargs={"heads": 2, "concat": False},  # Custom kwargs
                ),
                MlpConfig(
                    num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear
                ),
            ],
            intermediate_sizes=[5],
        )

        experiment_config.max_n_iters = 1
        experiment_config.evaluation = False

        task = VmasTask.BALANCE.get_from_yaml()
        experiment = Experiment(
            algorithm_config=MappoConfig.get_from_yaml(),
            model_config=gnn_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
