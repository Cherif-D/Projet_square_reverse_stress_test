from __future__ import annotations

from src.model.engine import ReverseStressEngine
from src.model.vasicek import tail_default_prob


def build_loss_rwa_ratio_fn(engine: ReverseStressEngine):
    """
    Backward-compatible wrapper around the central quantitative engine.

    The optimization pipeline and figures now use the same engine directly.
    """
    if not isinstance(engine, ReverseStressEngine):
        raise TypeError("build_loss_rwa_ratio_fn attend un ReverseStressEngine")
    return engine.evaluate
