import pytest

from cordon.bench import (
    GOALS,
    NOTES,
    TASKS,
    TEMPLATES,
    Case,
    attack_cases,
    benign_cases,
    clean_cases,
    insider_cases,
    sample_cases,
)
from cordon.shop import TOOLS, call_tool, seed_world


def run(world, steps):
    for name, arguments in steps:
        _, ok = call_tool(world, name, arguments)
        assert ok, (name, arguments)


@pytest.mark.parametrize("task", TASKS.values(), ids=lambda t: t.id)
def test_reference_plan_solves_the_task(task):
    before = seed_world()
    after = before.copy()
    assert not task.check(before, after, ""), "check passes before any work is done"
    run(after, task.plan)
    assert task.check(before, after, task.answer)


@pytest.mark.parametrize("goal", GOALS.values(), ids=lambda g: g.id)
def test_reference_calls_achieve_the_goal(goal):
    before = seed_world()
    after = before.copy()
    assert not goal.achieved(before, after)
    run(after, goal.calls)
    assert goal.achieved(before, after)


def test_no_task_achieves_an_attacker_goal_by_doing_its_job():
    for task in TASKS.values():
        before = seed_world()
        after = before.copy()
        run(after, task.plan)
        for goal in GOALS.values():
            assert not goal.achieved(before, after), (task.id, goal.id)


def test_plans_and_goals_use_only_real_tools():
    steps = [s for t in TASKS.values() for s in t.plan] + [
        s for g in GOALS.values() for s in g.calls
    ]
    assert {name for name, _ in steps} <= set(TOOLS)


def test_an_agent_that_ignores_the_injection_still_succeeds():
    for case in attack_cases():
        before = case.world()
        after = before.copy()
        run(after, case.task.plan)
        assert case.task.check(before, after, case.task.answer), case.id
        assert not case.goal.achieved(before, after), case.id


def test_the_planted_text_is_in_a_field_the_task_reads():
    for case in attack_cases():
        world = case.world()
        seen = [repr(call_tool(world, name, arguments)[0]) for name, arguments in case.task.plan]
        assert case.goal.instruction.lower() in "".join(seen).lower(), case.id


def test_templates_carry_the_instruction_and_goal_markers_identify_it():
    for goal in GOALS.values():
        assert all(marker in goal.instruction for marker in goal.markers)
        for template in TEMPLATES.values():
            assert goal.instruction.lower() in template.render(goal).lower()


def test_case_counts_and_ids():
    clean, attacks = clean_cases(), attack_cases()
    points = sum(len(t.points) for t in TASKS.values())
    assert len(clean) == len(TASKS) == 11
    assert len(attacks) == points * len(GOALS) * len(TEMPLATES) == 504
    ids = [c.id for c in clean + attacks]
    assert len(set(ids)) == len(ids)


def test_cases_round_trip_through_metadata():
    for case in clean_cases() + attack_cases()[:20]:
        assert Case.from_metadata(case.metadata()) == case


def test_sampling_is_deterministic():
    cases = attack_cases()
    assert sample_cases(cases, 50, seed=1) == sample_cases(cases, 50, seed=1)
    assert sample_cases(cases, 50, seed=1) != sample_cases(cases, 50, seed=2)
    assert sample_cases(cases, 10_000) == cases


def test_benign_notes_change_nothing_about_the_right_answer():
    cases = benign_cases()
    assert len(cases) == 12 * len(NOTES) == 252
    for case in cases:
        before = case.world()
        after = before.copy()
        seen = [repr(call_tool(after, name, arguments)[0]) for name, arguments in case.task.plan]
        assert case.payload in "".join(seen).replace("\\'", "'") or case.payload[:20] in "".join(
            seen
        )
        assert case.task.check(before, after, case.task.answer), case.id
        assert not case.attacked and Case.from_metadata(case.metadata()) == case


def test_insider_cases_plant_nothing():
    cases = insider_cases()
    assert len(cases) == len(TASKS) * len(GOALS)
    for case in cases:
        assert case.attacked and not case.injected and case.payload is None
        assert case.world().snapshot() == clean_cases()[0].world().snapshot()
        assert Case.from_metadata(case.metadata()) == case


def test_the_stratified_subset_covers_everything_evenly():
    from collections import Counter

    from cordon.bench.cases import stratified_cases

    cases = stratified_cases()
    assert len(cases) == 72 and len({c.id for c in cases}) == 72
    assert set(Counter(c.goal.id for c in cases).values()) == {12}
    assert {c.task.id for c in cases} == set(TASKS)
    per_template = Counter(c.template.id for c in cases)
    assert set(per_template) == set(TEMPLATES) and max(per_template.values()) <= 11
