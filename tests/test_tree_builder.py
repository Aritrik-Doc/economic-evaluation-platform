import pytest

from model.tree_builder import BuilderValidationError, compile_builder_tables, parse_reward_list


def parameter_rows():
    base={"source_url":"","publication_year":None,"assumption":"Test assumption","assumption_rationale":"Unit test","uncertainty_kind":"none","uncertainty_rationale":"Fixed for test","lower":None,"upper":None,"currency":"","price_year":None,"cost_bearers":""}
    return [
        {**base,"id":"p","label":"Probability","value":0.7,"unit":"proportion","category":"clinical","source_type":"user_assumption","source_citation":"Test source"},
        {**base,"id":"cost","label":"Cost","value":100,"unit":"GBP","category":"cost","source_type":"tariff","source_citation":"Test tariff","publication_year":2026,"currency":"GBP","price_year":2026,"cost_bearers":"health_system"},
        {**base,"id":"qaly","label":"QALY","value":1,"unit":"QALY","category":"utility","source_type":"randomised_trial","source_citation":"Test trial","publication_year":2026},
    ]


def strategy_rows():
    return [{"strategy_id":"A","strategy_name":"A","root_node_id":"root"},{"strategy_id":"B","strategy_name":"B","root_node_id":"terminal"}]


def node_rows():
    return [
        {"id":"root","label":"Root","type":"chance","cost_rewards":"cost@0","outcome_rewards":""},
        {"id":"terminal","label":"Terminal","type":"terminal","cost_rewards":"","outcome_rewards":"qaly@1"},
    ]


def branch_rows():
    return [
        {"from_node":"root","label":"Yes","probability_parameter_id":"p","probability_mode":"direct","to_node":"terminal"},
        {"from_node":"root","label":"No","probability_parameter_id":"p","probability_mode":"complement","to_node":"terminal"},
    ]


def test_builder_compiles_timed_rewards_and_complements():
    compiled=compile_builder_tables(parameter_rows(),strategy_rows(),node_rows(),branch_rows())
    root=compiled.tree.chance_nodes[0]
    terminal=compiled.tree.terminal_nodes[0]
    assert root.cost_rewards[0].time_years == 0
    assert terminal.outcome_rewards[0].time_years == 1
    assert root.branches[1].probability_mode == "complement"


def test_parse_reward_list_defaults_to_time_zero():
    rewards=parse_reward_list("a, b@2.5")
    assert [(r.parameter_id,r.time_years) for r in rewards] == [("a",0),("b",2.5)]


def test_invalid_reward_time_rejected():
    with pytest.raises(BuilderValidationError,match="invalid time"):
        parse_reward_list("cost@later")
