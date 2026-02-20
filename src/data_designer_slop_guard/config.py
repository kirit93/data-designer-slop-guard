from __future__ import annotations

from typing import Literal

from pydantic import Field

from data_designer.config.column_configs import SingleColumnConfig


class SlopGuardColumnConfig(SingleColumnConfig):
    """Score text columns for formulaic AI writing patterns using regex-based analysis.

    Runs ~80 compiled patterns against each row's text and produces a numeric score
    (0-100), a severity band, and actionable advice for each detected violation.

    Attributes:
        target_columns: Columns whose text content will be concatenated and scored.
        min_score: Minimum slop score (0-100) for ``is_valid=True``. Defaults to 60
            (the boundary between "light" and "moderate" slop).
        include_advice: Include per-violation actionable advice strings in output.
        include_violations: Include raw violation details (rule, match, context, penalty).
    """

    target_columns: list[str]
    min_score: int = Field(default=60, ge=0, le=100, description="Minimum slop score for is_valid=True")
    include_advice: bool = Field(default=True, description="Include actionable advice strings in output")
    include_violations: bool = Field(default=False, description="Include raw violation details in output")
    column_type: Literal["slop-guard"] = "slop-guard"

    @staticmethod
    def get_column_emoji() -> str:
        return "\U0001f9f9"

    @property
    def required_columns(self) -> list[str]:
        return self.target_columns

    @property
    def side_effect_columns(self) -> list[str]:
        return []
