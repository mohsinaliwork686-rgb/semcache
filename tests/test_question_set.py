"""The labelled question set follows the rules in docs/IMPLEMENTATION_PLAN.md, section 5a."""

import json
import re
from pathlib import Path

import pytest

QUESTIONS = Path(__file__).parent.parent / "bench" / "questions.json"
TOPICS = {"orders", "refunds", "shipping", "accounts", "payments", "product-care"}


@pytest.fixture(scope="module")
def data():
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def intents(data):
    return {intent["id"]: intent for intent in data["intents"]}


def test_about_50_intents_and_250_questions(data, intents):
    assert len(intents) == len(data["intents"]), "intent ids must be unique"
    assert 45 <= len(intents) <= 55
    assert 225 <= sum(len(i["questions"]) for i in intents.values()) <= 275


def test_each_intent_has_a_plain_question_and_3_to_5_rewordings(intents):
    for intent in intents.values():
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", intent["id"])
        assert intent["topic"] in TOPICS, intent["id"]
        assert 4 <= len(intent["questions"]) <= 6, intent["id"]
        assert all(q.strip() == q and q for q in intent["questions"]), intent["id"]


def test_different_intents_have_different_answers(intents):
    # This is what makes a match across intents a false hit.
    answers = [intent["answer"] for intent in intents.values()]
    assert all(answers)
    assert len(set(answers)) == len(answers)


def test_no_question_appears_twice(intents):
    seen = {}
    for intent in intents.values():
        for question in intent["questions"]:
            text = " ".join(question.lower().split())
            assert text not in seen, f"{question!r} is in {seen.get(text)} and {intent['id']}"
            seen[text] = intent["id"]


def test_near_misses_come_in_pairs_of_the_same_kind(data, intents):
    kinds = data["near_miss_kinds"]
    assert set(kinds) == {"amount", "negation", "swapped", "timing", "same-verb"}
    for intent in intents.values():
        near_miss = intent.get("near_miss")
        if near_miss is None:
            continue
        assert near_miss["kind"] in kinds, intent["id"]
        other = intents[near_miss["pair"]]
        assert other["near_miss"] == {"kind": near_miss["kind"], "pair": intent["id"]}


def test_every_kind_of_near_miss_has_several_pairs(intents):
    pairs_per_kind = {}
    for intent in intents.values():
        if "near_miss" in intent:
            kind = intent["near_miss"]["kind"]
            pairs_per_kind[kind] = pairs_per_kind.get(kind, 0) + 0.5
    assert all(count >= 3 for count in pairs_per_kind.values()), pairs_per_kind
