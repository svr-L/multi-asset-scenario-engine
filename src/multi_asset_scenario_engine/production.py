"""Validated production policy for the monthly physical-measure scenario engine.

V33 rejects regime-conditioned switching and does not validate task-specific
overrides against the universal ex-ante engine. The conservative production
policy is therefore one fixed 1M scenario law.
"""
from __future__ import annotations
from dataclasses import dataclass

VALIDATED_HORIZON = "1M"
HORIZON_BUSINESS_DAYS = 21
PRODUCTION_MODEL = "raw_duration|hierarchical|direct_H"
DRIVER_COLUMNS = (
    "EQ_EW", "RATE_DGS2", "RATE_DGS10", "RATE_DGS30",
    "CREDIT_Aaa_spr", "CREDIT_Baa_spr", "FX_EURUSD",
)

@dataclass(frozen=True)
class ProductionPolicy:
    horizon: str = VALIDATED_HORIZON
    horizon_business_days: int = HORIZON_BUSINESS_DAYS
    model: str = PRODUCTION_MODEL
    measure: str = "P"
    driver_columns: tuple[str, ...] = DRIVER_COLUMNS

def validated_policy() -> ProductionPolicy:
    return ProductionPolicy()

def model_for(objective: str | None = None) -> str:
    """Return the validated model. `objective` is API-compatible but non-operative."""
    return PRODUCTION_MODEL
