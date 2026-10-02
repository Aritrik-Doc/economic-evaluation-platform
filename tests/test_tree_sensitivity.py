import pytest

from model.decision_tree import ChanceNode, DecisionTreeDefinition, StrategyRoot, TerminalNode, TreeBranch
from model.schema import AssumptionSpec,EvidenceSource,Parameter,UncertaintySpec
from model.sensitivity import OneWaySensitivitySpec,TwoWaySensitivitySpec,ThresholdAnalysisSpec
from model.tree_sensitivity import one_way_tree_inmb,two_way_tree_inmb,threshold_tree_inmb

SOURCE=EvidenceSource("Test","user_assumption")
ASSUMPTION=AssumptionSpec("Test","Test")
U=UncertaintySpec("none","Test")

def p(id,value,category="clinical"):
    kw={}
    if category=="cost": kw={"currency":"GBP","price_year":2026,"cost_bearers":("health_system",)}
    return Parameter(id,id,value,"unit",category,SOURCE,ASSUMPTION,U,**kw)

def model():
    params=(p("prob",0.5),p("cost_a",100,"cost"),p("cost_b",0,"cost"),p("good",2,"utility"),p("bad",0,"utility"))
    tree=DecisionTreeDefinition(
        (StrategyRoot("A","root"),StrategyRoot("B","b")),
        (ChanceNode("root","Root",(TreeBranch("Good","prob","good"),TreeBranch("Bad","prob","bad","complement")),cost_parameter_ids=("cost_a",)),),
        (TerminalNode("good","Good",outcome_parameter_ids=("good",)),TerminalNode("bad","Bad",outcome_parameter_ids=("bad",)),TerminalNode("b","B",cost_parameter_ids=("cost_b",),outcome_parameter_ids=("bad",))),
    )
    return tree,params

def context():
    return dict(intervention_id="A",comparator_id="B",willingness_to_pay=100,included_cost_bearers=("health_system",))

def test_one_way_inmb():
    tree,params=model()
    result=one_way_tree_inmb(tree,params,OneWaySensitivitySpec("prob",(0.25,0.5,0.75)),**context())
    assert [row[0] for row in result] == [0.25,0.5,0.75]
    assert [row[1] for row in result] == pytest.approx([-50,0,50])

def test_two_way_inmb():
    tree,params=model()
    result=two_way_tree_inmb(tree,params,TwoWaySensitivitySpec("prob",(0.5,1.0),"cost_a",(100,150)),**context())
    assert len(result)==4
    assert result[0][2] == pytest.approx(0)
    assert result[-1][2] == pytest.approx(50)

def test_threshold_inmb():
    tree,params=model()
    value=threshold_tree_inmb(tree,params,ThresholdAnalysisSpec("prob",0,1),**context())
    assert value == pytest.approx(0.5,abs=1e-5)
