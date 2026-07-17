"""End-to-end runs with scripted models, whose behaviour is known in advance."""

from inspect_ai import eval

import cordon._registry  # noqa: F401  registers the scripted models
from cordon.bench import TEMPLATES, Case, attack_cases
from cordon.bench.eval import shop_attacks, shop_clean


def run(task, model, tmp_path):
    (log,) = eval(task, model=f"scripted/{model}", log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    return {s.id: s for s in log.samples}


def rate(samples, key):
    values = [s.scores["shop_scorer"].value[key] for s in samples.values()]
    return sum(values) / len(values)


def test_oracle_solves_every_clean_task(tmp_path):
    samples = run(shop_clean(), "oracle", tmp_path)
    assert len(samples) == 11 and rate(samples, "utility") == 1.0


def test_oracle_is_never_hijacked(tmp_path):
    samples = run(shop_attacks(cases=60), "oracle", tmp_path)
    assert rate(samples, "utility") == 1.0 and rate(samples, "attack") == 0.0


def test_gullible_is_always_hijacked_and_still_does_its_job(tmp_path):
    samples = run(shop_attacks(cases=60), "gullible", tmp_path)
    assert rate(samples, "attack") == 1.0 and rate(samples, "utility") == 1.0
    assert all(s.scores["shop_scorer"].metadata["attempted"] for s in samples.values())


def attacks_won(samples):
    return {s.id for s in samples.values() if s.scores["shop_scorer"].value["attack"]}


def test_taint_stops_everything_but_the_reply(tmp_path):
    samples = run(shop_attacks(defence="taint", cases=120), "gullible", tmp_path)
    # Replying to a ticket is not a high-risk tool, so the phishing reply gets through.
    assert {samples[i].metadata["goal"] for i in attacks_won(samples)} == {"phish"}


def test_grounding_stops_targets_named_only_by_the_attacker(tmp_path):
    samples = run(shop_attacks(defence="grounding", cases=120), "gullible", tmp_path)
    won = {samples[i].metadata["goal"] for i in attacks_won(samples)}
    assert not won & {"refund", "exfiltrate", "discount", "phish"}


def test_least_privilege_stops_attacks_that_need_a_tool_the_task_lacks(tmp_path):
    samples = run(shop_attacks(defence="least_privilege", cases=120), "gullible", tmp_path)
    won = attacks_won(samples)
    assert 0 < len(won) < len(samples)
    for sample_id in won:
        case = Case.from_metadata(samples[sample_id].metadata)
        assert {name for name, _ in case.goal.calls} <= case.task.tools


def test_taint_costs_usefulness_and_grounding_does_not(tmp_path):
    assert rate(run(shop_clean(defence="taint"), "oracle", tmp_path), "utility") < 1.0
    assert rate(run(shop_clean(defence="grounding"), "oracle", tmp_path), "utility") == 1.0
    assert rate(run(shop_clean(defence="least_privilege"), "oracle", tmp_path), "utility") == 1.0


def test_datamark_hides_the_instruction_from_a_string_matcher(tmp_path):
    samples = run(shop_attacks(defence="datamark", cases=40), "gullible", tmp_path)
    assert rate(samples, "attack") == 0.0


def test_blocked_calls_are_counted(tmp_path):
    samples = run(shop_attacks(defence="taint", cases=40), "gullible", tmp_path)
    assert any(s.scores["shop_scorer"].metadata["blocked"] for s in samples.values())


def test_subset_is_drawn_from_the_full_matrix():
    ids = {s.id for s in shop_attacks(cases=25).dataset}
    assert len(ids) == 25 and ids <= {c.id for c in attack_cases()}
    assert len(TEMPLATES) == 7
