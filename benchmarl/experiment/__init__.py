#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Experiment orchestration and training loop management.

Provides the Experiment class for running MARL training runs with support for
checkpointing, evaluation, callbacks, and logging to various backends.
"""

from .callback import Callback
from .experiment import Experiment, ExperimentConfig
