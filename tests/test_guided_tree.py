import pytest

from model.guided_tree import (
    GuidedTreeError,
    add_branch_with_child,
    add_strategy,
    append_reward,
    delete_leaf_node,
)


def test_add_strategy_creates_root_chance_node():
    strategies, nodes = add_strategy([], [], strategy_name="New treatment")
    assert strategies[0]["strategy_id"] == "new_treatment"
    assert nodes[0]["id"] == strategies[0]["root_node_id"]
    assert nodes[0]["type"] == "chance"


def test_add_branch_creates_child_and_constrained_edge():
    strategies, nodes = add_strategy([], [], strategy_name="Treatment")
    nodes, branches, child_id = add_branch_with_child(
        nodes,
        [],
        parent_id=strategies[0]["root_node_id"],
        branch_label="Responds",
        probability_parameter_id="p_response",
        probability_mode="direct",
        child_label="Response",
        child_type="terminal",
    )
    assert any(row["id"] == child_id and row["type"] == "terminal" for row in nodes)
    assert branches[0]["to_node"] == child_id


def test_branch_cannot_start_from_terminal_node():
    nodes = [{"id": "end", "label": "End", "type": "terminal"}]
    with pytest.raises(GuidedTreeError, match="chance"):
        add_branch_with_child(
            nodes,
            [],
            parent_id="end",
            branch_label="More",
            probability_parameter_id="p",
            probability_mode="direct",
            child_label="Next",
            child_type="terminal",
        )


def test_append_reward_uses_timed_reward_syntax():
    nodes = [{"id": "end", "label": "End", "type": "terminal", "cost_rewards": "", "outcome_rewards": ""}]
    updated = append_reward(nodes, "end", parameter_id="cost_followup", time_years=2.5, reward_kind="cost")
    assert updated[0]["cost_rewards"] == "cost_followup@2.5"


def test_delete_leaf_removes_incoming_branch():
    strategies = [{"strategy_id": "a", "strategy_name": "A", "root_node_id": "root"}]
    nodes = [
        {"id": "root", "label": "Event", "type": "chance"},
        {"id": "end", "label": "End", "type": "terminal"},
    ]
    branches = [{"from_node": "root", "to_node": "end", "label": "Outcome", "probability_parameter_id": "p", "probability_mode": "direct"}]
    new_nodes, new_branches = delete_leaf_node(strategies, nodes, branches, "end")
    assert [row["id"] for row in new_nodes] == ["root"]
    assert new_branches == []


def test_root_cannot_be_deleted():
    strategies = [{"strategy_id": "a", "strategy_name": "A", "root_node_id": "root"}]
    with pytest.raises(GuidedTreeError, match="root"):
        delete_leaf_node(strategies, [{"id": "root", "type": "terminal"}], [], "root")
