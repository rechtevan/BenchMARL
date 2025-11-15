#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Benchmark orchestration for running multiple MARL experiments.

Provides the Benchmark class for comparing different algorithms, tasks, and
models across multiple seeds with parallel execution support.
"""

from .benchmark import Benchmark
