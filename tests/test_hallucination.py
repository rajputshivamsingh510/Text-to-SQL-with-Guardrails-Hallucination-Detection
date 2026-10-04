from app import hallucination as hal
from app.db import QueryResult
from app.guardrails.sql_guard import validate_sql


def _grounding(sql, schema, engine):
    check = validate_sql(sql, schema, "sqlite")
    assert check.ok, check.feedback()
    return hal.check_value_grounding(check.qualified, engine, schema)


def test_existing_filter_value_passes(schema, engine):
    s = _grounding("SELECT name FROM customers WHERE country = 'India'", schema, engine)
    assert s.passed and s.score == 1.0 and not s.hard_fail


def test_hallucinated_filter_value_is_caught_with_hint(schema, engine):
    s = _grounding("SELECT name FROM customers WHERE country = 'USA'", schema, engine)
    assert s.hard_fail and s.score == 0
    assert "'USA'" in s.detail and "United States" in s.detail   # tells the LLM the real values


def test_wrong_casing_is_caught(schema, engine):
    s = _grounding("SELECT name FROM customers WHERE country = 'india'", schema, engine)
    assert s.hard_fail and "casing" in s.detail


def test_in_list_and_aliases_are_checked(schema, engine):
    s = _grounding("SELECT o.id FROM orders o WHERE o.status IN ('completed', 'refunded')", schema, engine)
    assert s.hard_fail and "refunded" in s.detail and "completed" not in s.detail.split("exist")[0]


def test_no_literals_means_signal_not_applicable(schema, engine):
    s = _grounding("SELECT COUNT(*) FROM orders", schema, engine)
    assert s.score is None and s.passed


def test_result_sanity():
    assert hal.check_result_sanity(QueryResult(["a"], [[1]], False)).score == 1.0
    assert hal.check_result_sanity(QueryResult(["a"], [], False)).score == 0.5
    assert hal.check_result_sanity(QueryResult(["a"], [[1]], True)).score == 0.9


def test_answer_grounding_accepts_real_numbers_and_formats():
    r = QueryResult(["country", "revenue"], [["India", 12345.5], ["Canada", 800]], False)
    ok = hal.check_answer_grounding("India leads with 12,345.50 in revenue, ahead of Canada at 800.", "revenue by country", r)
    assert ok.passed and ok.score == 1.0
    assert hal.check_answer_grounding("There are 2 countries.", "q", r).passed   # row count is allowed


def test_answer_grounding_catches_invented_numbers():
    r = QueryResult(["country", "revenue"], [["India", 100], ["Canada", 50]], False)
    bad = hal.check_answer_grounding("India made 100, Canada 50, a total of 150.", "q", r)  # 150 was computed, not returned
    assert not bad.passed and "150" in bad.detail


def test_self_consistency():
    main = QueryResult(["n"], [[5]], False)
    same = QueryResult(["count"], [[5]], False)
    diff = QueryResult(["n"], [[9]], False)
    assert hal.check_self_consistency(main, [same, same]).score == 1.0
    assert hal.check_self_consistency(main, [diff, diff]).hard_fail          # alternatives agree with each other, not with main
    assert not hal.check_self_consistency(main, [diff, QueryResult(["n"], [[7]], False)]).hard_fail
    assert hal.check_self_consistency(main, [None]).score is None


def test_combine_weights_hard_fail_and_threshold():
    good = [hal.Signal("value_grounding", 1.0, True, ""), hal.Signal("semantic_judge", 0.9, True, "")]
    assert hal.combine(good, 0.6).passed and hal.combine(good, 0.6).label == "high"
    hard = good + [hal.Signal("result_sanity", 0.0, False, "bad", hard_fail=True)]
    assert not hal.combine(hard, 0.6).passed
    low = [hal.Signal("semantic_judge", 0.4, False, "meh")]
    assert not hal.combine(low, 0.6).passed
    assert hal.combine([], 0.6).confidence == 0.5
