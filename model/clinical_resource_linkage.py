"""Natural-unit resource linkage from clinical models to capacity planning.

Clinical models supply disease/event trajectories per patient. This module maps
explicit ``resource_use`` parameters onto states or decision-tree nodes and
projects annual natural-unit profiles for capacity planning. Costs are never
interpreted as physical resources automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isclose
from typing import Literal, Mapping, Sequence

from model.decision_tree import ChanceNode, DecisionTreeDefinition, validate_decision_tree
from model.markov import CohortMarkovDefinition, run_cohort_markov
from model.resource_capacity import ResourceRequirement
from model.schema import Parameter
from model.semi_markov import SemiMarkovDefinition, run_semi_markov


ResourceAccrual = Literal["per_year", "per_cycle"]


class ClinicalResourceLinkageError(ValueError):
    """Raised when a clinical resource linkage is incomplete or incoherent."""


@dataclass(frozen=True)
class StateResourceMapping:
    strategy_id: str
    state_id: str
    resource_id: str
    parameter_id: str
    accrual: ResourceAccrual = "per_year"

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.strategy_id,
                self.state_id,
                self.resource_id,
                self.parameter_id,
            )
        ):
            raise ValueError(
                "State resource mappings require strategy, state, resource and parameter ids."
            )
        if self.accrual not in {"per_year", "per_cycle"}:
            raise ValueError("Resource accrual must be per_year or per_cycle.")


@dataclass(frozen=True)
class TreeResourceMapping:
    strategy_id: str
    node_id: str
    resource_id: str
    parameter_id: str
    time_years: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.strategy_id,
                self.node_id,
                self.resource_id,
                self.parameter_id,
            )
        ):
            raise ValueError(
                "Tree resource mappings require strategy, node, resource and parameter ids."
            )
        if self.time_years < 0:
            raise ValueError("Decision-tree resource time cannot be negative.")


@dataclass(frozen=True)
class ClinicalResourceProfile:
    strategy_id: str
    strategy_name: str
    resource_id: str
    annual_units_per_patient: tuple[float, ...]
    source_model_type: str
    included_parameter_ids: tuple[str, ...]


def available_resource_parameters(parameters: Sequence[Parameter]) -> tuple[str, ...]:
    return tuple(
        sorted(parameter.id for parameter in parameters if parameter.category == "resource_use")
    )


def _resource_value(
    parameter_map: Mapping[str, Parameter],
    parameter_id: str,
    overrides: Mapping[str, float] | None,
) -> float:
    parameter = parameter_map.get(parameter_id)
    if parameter is None:
        raise ClinicalResourceLinkageError(
            f"Resource mapping references undefined parameter '{parameter_id}'."
        )
    if parameter.category != "resource_use":
        raise ClinicalResourceLinkageError(
            f"Resource mapping parameter '{parameter_id}' must use category 'resource_use', "
            f"not '{parameter.category}'."
        )
    value = float((overrides or {}).get(parameter_id, parameter.value))
    if value < 0:
        raise ClinicalResourceLinkageError(
            "Resource-use parameter values must be non-negative."
        )
    return value


def _cycles_per_year(cycle_length_years: float) -> int:
    if cycle_length_years <= 0 or cycle_length_years > 1:
        raise ClinicalResourceLinkageError(
            "Clinical resource linkage currently requires a cycle length greater than 0 "
            "and no longer than 1 year."
        )
    reciprocal = 1.0 / cycle_length_years
    rounded = int(round(reciprocal))
    if rounded < 1 or not isclose(
        reciprocal, rounded, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ClinicalResourceLinkageError(
            "For annual capacity planning, the clinical cycle length must divide one year "
            "exactly (for example 1, 1/2, 1/4 or 1/12 year)."
        )
    return rounded


def _state_weight(start: float, end: float, timing: str) -> float:
    if timing == "start":
        return start
    if timing == "end":
        return end
    return 0.5 * (start + end)


def _project_state_profiles(
    *,
    model,
    run,
    parameters: Sequence[Parameter],
    mappings: Sequence[StateResourceMapping],
    horizon_years: int,
    source_model_type: str,
    overrides: Mapping[str, float] | None,
) -> tuple[ClinicalResourceProfile, ...]:
    cycles_per_year = _cycles_per_year(model.cycle_length_years)
    required_cycles = horizon_years * cycles_per_year
    if model.max_cycles < required_cycles:
        raise ClinicalResourceLinkageError(
            "The clinical model horizon is shorter than the requested resource-planning horizon. "
            "Extend the clinical model rather than assuming zero resource use beyond its horizon."
        )

    parameter_map = {parameter.id: parameter for parameter in parameters}
    strategy_ids = {strategy.strategy_id for strategy in model.strategies}
    state_ids = {state.id for state in model.states}
    if not mappings:
        raise ClinicalResourceLinkageError(
            "Add at least one explicit clinical resource mapping."
        )
    for mapping in mappings:
        if mapping.strategy_id not in strategy_ids:
            raise ClinicalResourceLinkageError(
                f"Resource mapping references unknown strategy '{mapping.strategy_id}'."
            )
        if mapping.state_id not in state_ids:
            raise ClinicalResourceLinkageError(
                f"Resource mapping references unknown state '{mapping.state_id}'."
            )
        _resource_value(parameter_map, mapping.parameter_id, overrides)

    mapping_by_strategy: dict[str, list[StateResourceMapping]] = {}
    for mapping in mappings:
        mapping_by_strategy.setdefault(mapping.strategy_id, []).append(mapping)

    profiles: list[ClinicalResourceProfile] = []
    for result in run.strategies:
        strategy_mappings = mapping_by_strategy.get(result.strategy_id, [])
        if not strategy_mappings:
            continue
        state_index = {
            state_id: index for index, state_id in enumerate(result.state_ids)
        }
        resources = sorted(
            {mapping.resource_id for mapping in strategy_mappings}
        )
        completed_cycles = max(0, len(result.trace) - 1)
        for resource_id in resources:
            relevant = [
                mapping
                for mapping in strategy_mappings
                if mapping.resource_id == resource_id
            ]
            cycle_units = [0.0] * required_cycles
            # If a model legitimately stops early under its configured cohort-depletion
            # rule, resource accrual stops with the model. Later planning periods remain
            # zero rather than indexing beyond the completed clinical trace.
            for cycle in range(min(required_cycles, completed_cycles)):
                start = result.trace[cycle]
                end = result.trace[cycle + 1]
                total = 0.0
                for mapping in relevant:
                    index = state_index[mapping.state_id]
                    exposure = _state_weight(
                        float(start[index]),
                        float(end[index]),
                        model.state_accrual_timing,
                    )
                    value = _resource_value(
                        parameter_map,
                        mapping.parameter_id,
                        overrides,
                    )
                    scale = (
                        model.cycle_length_years
                        if mapping.accrual == "per_year"
                        else 1.0
                    )
                    total += exposure * value * scale
                cycle_units[cycle] = total
            annual = tuple(
                sum(
                    cycle_units[
                        index * cycles_per_year : (index + 1) * cycles_per_year
                    ]
                )
                for index in range(horizon_years)
            )
            profiles.append(
                ClinicalResourceProfile(
                    strategy_id=result.strategy_id,
                    strategy_name=result.label,
                    resource_id=resource_id,
                    annual_units_per_patient=annual,
                    source_model_type=source_model_type,
                    included_parameter_ids=tuple(
                        sorted({mapping.parameter_id for mapping in relevant})
                    ),
                )
            )
    if not profiles:
        raise ClinicalResourceLinkageError(
            "The selected clinical resource mappings produced no strategy profiles."
        )
    return tuple(profiles)


def project_markov_resource_profiles(
    model: CohortMarkovDefinition,
    parameters: Sequence[Parameter],
    *,
    mappings: Sequence[StateResourceMapping],
    horizon_years: int,
    overrides: Mapping[str, float] | None = None,
) -> tuple[ClinicalResourceProfile, ...]:
    run = run_cohort_markov(
        model,
        parameters,
        overrides=overrides,
        cost_discount_rate=0.0,
        outcome_discount_rate=0.0,
    )
    return _project_state_profiles(
        model=model,
        run=run,
        parameters=parameters,
        mappings=mappings,
        horizon_years=horizon_years,
        source_model_type="cohort_markov",
        overrides=overrides,
    )


def project_semi_markov_resource_profiles(
    model: SemiMarkovDefinition,
    parameters: Sequence[Parameter],
    *,
    mappings: Sequence[StateResourceMapping],
    horizon_years: int,
    overrides: Mapping[str, float] | None = None,
) -> tuple[ClinicalResourceProfile, ...]:
    run = run_semi_markov(
        model,
        parameters,
        overrides=overrides,
        cost_discount_rate=0.0,
        outcome_discount_rate=0.0,
    )
    return _project_state_profiles(
        model=model,
        run=run,
        parameters=parameters,
        mappings=mappings,
        horizon_years=horizon_years,
        source_model_type="semi_markov",
        overrides=overrides,
    )


def _budget_year(time_years: float) -> int:
    if time_years <= 0:
        return 1
    return max(1, int(ceil(time_years - 1e-12)))


def project_decision_tree_resource_profiles(
    tree: DecisionTreeDefinition,
    parameters: Sequence[Parameter],
    *,
    mappings: Sequence[TreeResourceMapping],
    horizon_years: int,
    strategy_names: Mapping[str, str] | None = None,
    overrides: Mapping[str, float] | None = None,
) -> tuple[ClinicalResourceProfile, ...]:
    if horizon_years < 1:
        raise ClinicalResourceLinkageError(
            "Resource linkage requires at least one annual period."
        )
    validate_decision_tree(tree, parameters, overrides=overrides)
    if not mappings:
        raise ClinicalResourceLinkageError(
            "Add at least one explicit decision-tree resource mapping."
        )

    parameter_map = {parameter.id: parameter for parameter in parameters}
    values = {
        parameter.id: float((overrides or {}).get(parameter.id, parameter.value))
        for parameter in parameters
    }
    chance = {node.id: node for node in tree.chance_nodes}
    terminal = {node.id: node for node in tree.terminal_nodes}
    roots = {
        root.strategy_id: root.root_node_id for root in tree.strategy_roots
    }
    for mapping in mappings:
        if mapping.strategy_id not in roots:
            raise ClinicalResourceLinkageError(
                f"Resource mapping references unknown strategy '{mapping.strategy_id}'."
            )
        if mapping.node_id not in chance and mapping.node_id not in terminal:
            raise ClinicalResourceLinkageError(
                f"Resource mapping references unknown node '{mapping.node_id}'."
            )
        _resource_value(parameter_map, mapping.parameter_id, overrides)

    mapping_lookup: dict[tuple[str, str], list[TreeResourceMapping]] = {}
    for mapping in mappings:
        mapping_lookup.setdefault(
            (mapping.strategy_id, mapping.node_id), []
        ).append(mapping)

    def branch_probability(branch) -> float:
        value = values[branch.probability_parameter_id]
        return value if branch.probability_mode == "direct" else 1.0 - value

    names = dict(strategy_names or {})
    profiles: list[ClinicalResourceProfile] = []
    for strategy_id, root_node_id in roots.items():
        resource_ids = sorted(
            {
                mapping.resource_id
                for mapping in mappings
                if mapping.strategy_id == strategy_id
            }
        )
        annual_by_resource = {
            resource_id: [0.0] * horizon_years
            for resource_id in resource_ids
        }
        parameter_ids_by_resource = {
            resource_id: set() for resource_id in resource_ids
        }

        def visit(node_id: str, reach_probability: float) -> None:
            node = terminal.get(node_id) or chance.get(node_id)
            if node is None:
                raise ClinicalResourceLinkageError(
                    f"Decision tree references unknown node '{node_id}'."
                )
            for mapping in mapping_lookup.get((strategy_id, node_id), []):
                year = _budget_year(mapping.time_years)
                if year <= horizon_years:
                    annual_by_resource[mapping.resource_id][year - 1] += (
                        reach_probability
                        * _resource_value(
                            parameter_map,
                            mapping.parameter_id,
                            overrides,
                        )
                    )
                    parameter_ids_by_resource[mapping.resource_id].add(
                        mapping.parameter_id
                    )
            if isinstance(node, ChanceNode):
                for branch in node.branches:
                    visit(
                        branch.child_node_id,
                        reach_probability * branch_probability(branch),
                    )

        visit(root_node_id, 1.0)
        for resource_id in resource_ids:
            profiles.append(
                ClinicalResourceProfile(
                    strategy_id=strategy_id,
                    strategy_name=names.get(strategy_id, strategy_id),
                    resource_id=resource_id,
                    annual_units_per_patient=tuple(
                        annual_by_resource[resource_id]
                    ),
                    source_model_type="decision_tree",
                    included_parameter_ids=tuple(
                        sorted(parameter_ids_by_resource[resource_id])
                    ),
                )
            )
    return tuple(profiles)


def profiles_to_requirements(
    profiles: Sequence[ClinicalResourceProfile],
    *,
    intervention_to_strategy: Mapping[str, str],
    horizon_years: int,
) -> tuple[ResourceRequirement, ...]:
    profile_map = {
        (profile.strategy_id, profile.resource_id): profile
        for profile in profiles
    }
    strategy_ids = {profile.strategy_id for profile in profiles}
    missing = set(intervention_to_strategy.values()) - strategy_ids
    if missing:
        raise ClinicalResourceLinkageError(
            "Intervention mapping references unavailable clinical strategy profile(s): "
            + ", ".join(sorted(missing))
        )
    rows: list[ResourceRequirement] = []
    resources = sorted({profile.resource_id for profile in profiles})
    for intervention_id, strategy_id in intervention_to_strategy.items():
        for resource_id in resources:
            profile = profile_map.get((strategy_id, resource_id))
            if profile is None:
                continue
            if len(profile.annual_units_per_patient) < horizon_years:
                raise ClinicalResourceLinkageError(
                    "Clinical resource profile is shorter than the planning horizon."
                )
            for period, value in enumerate(
                profile.annual_units_per_patient[:horizon_years],
                start=1,
            ):
                if value > 0:
                    rows.append(
                        ResourceRequirement(
                            intervention_id,
                            resource_id,
                            period,
                            value,
                        )
                    )
    return tuple(rows)


def combine_resource_requirements(
    manual: Sequence[ResourceRequirement],
    linked: Sequence[ResourceRequirement],
) -> tuple[ResourceRequirement, ...]:
    """Add manual and linked requirements for hybrid capacity planning."""
    totals: dict[tuple[str, str, int], float] = {}
    for row in (*tuple(manual), *tuple(linked)):
        key = (row.intervention_id, row.resource_id, row.period)
        totals[key] = totals.get(key, 0.0) + row.units_per_person
    return tuple(
        ResourceRequirement(
            intervention_id,
            resource_id,
            period,
            value,
        )
        for (intervention_id, resource_id, period), value in sorted(
            totals.items()
        )
        if value > 0
    )
