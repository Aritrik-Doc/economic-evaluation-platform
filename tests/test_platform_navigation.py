from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PAGE_PATTERN = re.compile(r'(?:st\.Page|st\.page_link)\(\s*["\'](pages/[^"\']+)["\']')


def _referenced_pages():
    files = [ROOT / "app.py", *(ROOT / "pages").glob("*.py")]
    references = set()
    for path in files:
        references.update(PAGE_PATTERN.findall(path.read_text(encoding="utf-8")))
    return references


def test_all_registered_and_linked_streamlit_pages_exist():
    references = _referenced_pages()
    assert references, "Expected at least one Streamlit page reference."
    missing = sorted(reference for reference in references if not (ROOT / reference).is_file())
    assert missing == []


def test_router_registers_all_stakeholder_facing_workspaces():
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    required = {
        "pages/0_Home.py",
        "pages/1_Decision_Tree_Builder.py",
        "pages/2_Cohort_Markov_Builder.py",
        "pages/3_Advanced_Markov_Dynamics.py",
        "pages/10_Population_Uptake.py",
        "pages/6_Budget_Impact_Analysis.py",
        "pages/7_BIA_Clinical_Linkage.py",
        "pages/8_Resource_Capacity_Planning.py",
        "pages/9_Policy_Interpretation.py",
        "pages/5_Transparency_Check.py",
        "pages/4_State_Transition_Save_Load_Audit.py",
    }
    assert required <= set(PAGE_PATTERN.findall(app_text))


def test_home_exposes_population_bia_and_capacity_directly():
    home = (ROOT / "pages/0_Home.py").read_text(encoding="utf-8")
    references = set(PAGE_PATTERN.findall(home))
    assert {
        "pages/10_Population_Uptake.py",
        "pages/6_Budget_Impact_Analysis.py",
        "pages/8_Resource_Capacity_Planning.py",
        "pages/9_Policy_Interpretation.py",
    } <= references
