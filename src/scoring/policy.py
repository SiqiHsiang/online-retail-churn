"""
Retention policy definition.

Policy version: v1.0
This module defines incentive cost, uplift assumptions,
and coupon assignment by customer value segment.
"""

POLICY_TABLE = {
    "low": {
        "cost": 5.0,
        "uplift": 0.05,
        "coupon_eur": 5,
    },
    "mid": {
        "cost": 5.0,
        "uplift": 0.10,
        "coupon_eur": 5,
    },
    "high": {
        "cost": 10.0,
        "uplift": 0.30,
        "coupon_eur": 10,
    },
}


def get_policy_params(value_segment: str) -> dict:
    """
    Return policy parameters for a given value segment.
    """
    if value_segment not in POLICY_TABLE:
        raise ValueError(f"Unknown value_segment: {value_segment}")
    return POLICY_TABLE[value_segment]