# SPDX-License-Identifier: Apache-2.0
"""Slop Guard plugin for NeMo Data Designer.

Adds a ``slop-guard`` column type that scores text for formulaic AI writing
patterns using ~80 compiled regex rules. No LLM calls, no API dependencies.

Usage::

    from data_designer_slop_guard import SlopGuardColumnConfig

    builder.add_column(SlopGuardColumnConfig(
        name="slop_check",
        target_columns=["article"],
        min_score=60,
    ))
"""

from data_designer_slop_guard.config import SlopGuardColumnConfig
from data_designer_slop_guard.core import Hyperparameters, analyze_text

__all__ = ["SlopGuardColumnConfig", "analyze_text", "Hyperparameters"]
