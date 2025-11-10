#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


"""VMAS Simple Crypto task configuration."""

from dataclasses import MISSING, dataclass
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for the VMAS Simple Crypto task.

    A communication task where agents must learn to encrypt and decrypt messages to
    coordinate reaching the correct landmark. Tests emergent communication and
    cryptographic-like coordination. VMAS implementation of the MPE Simple Crypto environment.

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        dim_c: Dimensionality of the communication channel.
    """

    max_steps: Any = MISSING
    dim_c: Any = MISSING
