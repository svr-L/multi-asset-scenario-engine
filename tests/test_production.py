from multi_asset_scenario_engine.production import DRIVER_COLUMNS, HORIZON_BUSINESS_DAYS, PRODUCTION_MODEL, model_for, validated_policy

def test_validated_monthly_policy_is_frozen():
    p = validated_policy()
    assert p.horizon == "1M"
    assert HORIZON_BUSINESS_DAYS == 21
    assert PRODUCTION_MODEL == "raw_duration|hierarchical|direct_H"
    assert len(DRIVER_COLUMNS) == 7
    assert model_for("joint_distribution") == PRODUCTION_MODEL
    assert model_for("tail_risk") == PRODUCTION_MODEL
