import pytest

from model.tree_builder import BuilderValidationError, compile_builder_tables


def parameter_rows():
    return [
        {"id":"p_a","label":"Probability A","value":0.7,"unit":"proportion","category":"clinical","source_type":"user_assumption","source_citation":"Test source","source_url":"","publication_year":None,"assumption":"Test assumption","assumption_rationale":"Unit test","uncertainty_kind":"none","uncertainty_rationale":"Fixed for test","lower":None,"upper":None,"currency":"","price_year":None,"cost_bearers":""},
        {"id":"p_b","label":"Probability B","value":0.3,"unit":"proportion","category":"clinical","source_type":"user_assumption","source_citation":"Test source","source_url":"","publication_year":None,"assumption":"Test assumption","assumption_rationale":"Unit test","uncertainty_kind":"none","uncertainty_rationale":"Fixed for test","lower":None,"upper":None,"currency":"","price_year":None,"cost_bearers":""},
        {"id":"cost","label":"Cost","value":100,"unit":"GBP","category":"cost","source_type":"tariff","source_citation":"Test tariff","source_url":"","publication_year":2026,"assumption":"Applies equally","assumption_rationale":"Unit test","uncertainty_kind":"none","uncertainty_rationale":"Fixed for test","lower":None,"upper":None,"currency":"GBP","price_year":2026,"cost_bearers":"health_system"},
        {"id":"qaly","label":"QALY","value":1,"unit":"QALY","category":"utility","source_type":"randomised_trial","source_citation":"Test trial","source_url":"","publication_year":2026,"assumption":"Applies to terminal outcome","assumption_rationale":"Unit test","uncertainty_kind":"none","uncertainty_rationale":"Fixed for test","lower":None,"upper":None,"currency":"","price_year":None,"cost_bearers":""},
    ]


def strategy_rows():
    return [
        {"strategy_id":"A","strategy_name":"A","root_node_id":"root_a"},
        {"strategy_id":"B","strategy_name":"B","root_node_id":"terminal"},
    ]


def node_rows():
    return [
        {"id":"root_a","label":"Root A","type":"chance","cost_parameter_ids":"","outcome_parameter_ids":""},
        {"id":"terminal","label":"Terminal","type":"terminal","cost_parameter_ids":"cost","outcome_parameter_ids":"qaly"},
    ]


def branch_rows():
    return [
        {"from_node":"root_a","label":"A","probability_parameter_id":"p_a","to_node":"terminal"},
        {"from_node":"root_a","label":"B","probability_parameter_id":"p_b","to_node":"terminal"},
    ]


def test_builder_compiles_tables():
    compiled=compile_builder_tables(parameter_rows(),strategy_rows(),node_rows(),branch_rows())
    assert len(compiled.parameters)==4
    assert len(compiled.tree.strategy_roots)==2
    assert compiled.strategy_names["A"]=="A"


def test_source_is_mandatory():
    rows=parameter_rows(); rows[0]["source_citation"]=""
    with pytest.raises(BuilderValidationError,match="source_citation"):
        compile_builder_tables(rows,strategy_rows(),node_rows(),branch_rows())


def test_terminal_node_cannot_have_outgoing_branch():
    branches=branch_rows()+[{"from_node":"terminal","label":"Invalid","probability_parameter_id":"p_a","to_node":"terminal"}]
    with pytest.raises(BuilderValidationError,match="cannot have outgoing"):
        compile_builder_tables(parameter_rows(),strategy_rows(),node_rows(),branches)
