from src.ext.cost_aware import (
    install_cost_provider, record_mapping, record_cost_by_id, cost_for_prompt
)

def test_cost_record_and_decay():
    # 50% EMA (decay = 0.5)
    install_cost_provider(metric="latency_ms", decay=0.5)

    record_mapping("Q0", 100)
    record_cost_by_id(100, 100.0)        # first observation
    assert abs(cost_for_prompt("Q0") - 100.0) < 1e-6

    record_cost_by_id(100, 50.0)         # EMA: 0.5*100 + 0.5*50 = 75
    assert abs(cost_for_prompt("Q0") - 75.0) < 1e-6

def test_missing_prompt_returns_zero():
    install_cost_provider(metric="latency_ms", decay=0.0)
    assert cost_for_prompt("does-not-exist") == 0.0
