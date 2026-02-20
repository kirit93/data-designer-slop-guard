from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from data_designer_slop_guard.config import SlopGuardColumnConfig
from data_designer_slop_guard.core import analyze_text

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class SlopGuardColumnGenerator(ColumnGeneratorFullColumn[SlopGuardColumnConfig]):
    """Column generator that scores text for AI slop patterns via regex analysis."""

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\U0001f9f9 Scoring column {self.config.name!r} for AI slop patterns")
        logger.info(f"   target columns: {self.config.target_columns}")
        logger.info(f"   min_score: {self.config.min_score}")

        results = []
        for _, row in data[self.config.target_columns].iterrows():
            text = " ".join(str(v) for v in row.values if v is not None)
            analysis = analyze_text(text)
            output: dict = {
                "is_valid": analysis["score"] >= self.config.min_score,
                "slop_score": analysis["score"],
                "slop_band": analysis["band"],
                "word_count": analysis["word_count"],
            }
            if self.config.include_advice:
                output["slop_advice"] = analysis["advice"]
            if self.config.include_violations:
                output["slop_violations"] = analysis["violations"]
                output["slop_counts"] = analysis["counts"]
            results.append(output)

        data = data.copy()
        data[self.config.name] = results
        return data
