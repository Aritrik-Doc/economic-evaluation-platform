import json

import pytest

from model.persistence import (
    PersistenceError,
    audit_trail_json,
    build_audit_record,
    build_decision_tree_bundle,
    load_decision_tree_bundle,
    model_bundle_json,
    rows_to_csv,
)


def parameter_rows():
    common = {
        "source_type":"user_assumption",
        "source_citation":"Test source",
        "source_url":"",
        "publication_year":None,
        "assumption":"Test assumption",
        "assumption_rationale":"Unit test",
        "uncertainty_kind":"none",
        "uncertainty_rationale":"Fixed for test",
        "lower":None,
        "upper":None,
        "distribution_family":"",
        "distribution_parameters":"",
        "correlation_group":"",
        "currency":"",
        "price_year":None,
        "cost_bearers":"",
        "notes":"",
    }
    return [
        {**common,"id":"p","label":"Probability","value":0.5,"unit":"proportion","category":"clinical"},
        {**common,"id":"cost","label":"Cost","value":100,"unit":"currency","category":"cost","currency":"GBP","price_year":2026,"cost_bearers":"health_system"},
        {**common,"id":"qaly","label":"QALY","value":1.0,"unit":"QALY","category":"utility"},
    ]


def strategy_rows():
    return [
        {"strategy_id":"A","strategy_name":"A","root_node_id":"root"},
        {"strategy_id":"B","strategy_name":"B","root_node_id":"b"},
    ]


def node_rows():
    return [
        {"id":"root","label":"Root","type":"chance","cost_rewards":"","outcome_rewards":""},
        {"id":"good","label":"Good","type":"terminal","cost_rewards":"cost@0","outcome_rewards":"qaly@1"},
        {"id":"bad","label":"Bad","type":"terminal","cost_rewards":"","outcome_rewards":""},
        {"id":"b","label":"B","type":"terminal","cost_rewards":"","outcome_rewards":""},
    ]


def branch_rows():
    return [
        {"from_node":"root","label":"Good","probability_parameter_id":"p","probability_mode":"direct","to_node":"good"},
        {"from_node":"root","label":"Bad","probability_parameter_id":"p","probability_mode":"complement","to_node":"bad"},
    ]


def bundle():
    return build_decision_tree_bundle(
        model_name="Test decision tree",
        reference_case_code="NICE_TA",
        outcome_code="QALY",
        currency_code="GBP",
        threshold=30000,
        perspective_label="NHS and PSS",
        included_cost_bearers=("health_system", "personal_social_services"),
        time_horizon="5 years",
        cost_discount_rate=0.035,
        outcome_discount_rate=0.035,
        parameter_rows=parameter_rows(),
        strategy_rows=strategy_rows(),
        node_rows=node_rows(),
        branch_rows=branch_rows(),
        author="Analyst",
        notes="Test model",
    )


def test_model_bundle_round_trip_and_hash_validation():
    original = bundle()
    loaded = load_decision_tree_bundle(model_bundle_json(original))
    assert loaded["model_name"] == "Test decision tree"
    assert loaded["methods"]["threshold"] == 30000
    assert loaded["content_hash_sha256"] == original["content_hash_sha256"]


def test_tampered_model_file_is_rejected():
    original = bundle()
    tampered = json.loads(model_bundle_json(original))
    tampered["methods"]["threshold"] = 99999
    with pytest.raises(PersistenceError, match="content hash"):
        load_decision_tree_bundle(json.dumps(tampered))


def test_invalid_structure_is_not_saved():
    bad_nodes = node_rows()
    bad_nodes[0]["id"] = "missing_root"
    with pytest.raises(Exception):
        build_decision_tree_bundle(
            model_name="Bad model",
            reference_case_code="NICE_TA",
            outcome_code="QALY",
            currency_code="GBP",
            threshold=30000,
            perspective_label="NHS and PSS",
            included_cost_bearers=("health_system",),
            time_horizon="5 years",
            cost_discount_rate=0.035,
            outcome_discount_rate=0.035,
            parameter_rows=parameter_rows(),
            strategy_rows=strategy_rows(),
            node_rows=bad_nodes,
            branch_rows=branch_rows(),
        )


def test_audit_record_embeds_model_snapshot_and_settings():
    model = bundle()
    record = build_audit_record(
        model,
        analysis_type="psa",
        run_settings={"iterations": 1000, "seed": 42},
        results={"probability_cost_effective": 0.72},
        warnings=("Correlation sampled independently.",),
    )
    assert record["analysis_type"] == "psa"
    assert record["run_settings"]["seed"] == 42
    assert record["model_hash_sha256"] == model["content_hash_sha256"]
    assert record["model_snapshot"]["model_name"] == "Test decision tree"
    assert record["warnings"] == ["Correlation sampled independently."]

    exported = json.loads(audit_trail_json([record]))
    assert len(exported["records"]) == 1
    assert exported["records"][0]["run_id"] == record["run_id"]


def test_rows_to_csv_is_human_readable():
    text = rows_to_csv([
        {"Strategy":"A","Cost":100,"QALY":1.2},
        {"Strategy":"B","Cost":80,"QALY":1.0},
    ])
    assert "Strategy,Cost,QALY" in text
    assert "A,100,1.2" in text
