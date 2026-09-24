"""Offline guards for the packaged prompt and fixed language-check definitions.

These are not model-behavior tests. The active policy is English-only. The old
v1 multilingual cases are retained as historical fixture definitions, not active
expectations or measured model results. No LLM is called.
"""

import json
from importlib.resources import files
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "evals/scenarios/response-language-v1.json"
EXPECTED_LANGUAGES = {
    "english_with_hindi_script_name": ["en"],
    "english_with_noisy_hindi_script_prefix": ["en"],
    "hindi_question": ["hi"],
    "mixed_hindi_english_question": ["hinglish"],
    "explicit_english_persists": ["en", "en"],
    "explicit_hindi_persists": ["hi", "hi"],
    "explicit_hinglish_persists": ["hinglish", "hinglish"],
    "explicit_preference_can_change": ["en", "hi", "hi"],
    "automatic_choice_follows_clear_changes": ["en", "hi", "en"],
    "quoted_instruction_is_not_a_preference": ["en"],
}


def read_prompt():
    return files("healthcare_voice_agent.agent").joinpath("prompts.md").read_text("utf-8")


def test_packaged_prompt_is_english_only_without_multilingual_directions():
    prompt = " ".join(read_prompt().split())
    assert "Always respond in English." in prompt
    assert "Do not switch languages or scripts" in prompt
    assert "Do not translate or transliterate" in prompt
    assert "in English rather than guessing" in prompt
    assert all(term not in prompt.casefold() for term in ("hindi", "hinglish", "devanagari", "code-switch"))


def test_language_change_preserves_demo_safety_and_brief_replies():
    prompt = " ".join(read_prompt().split())
    assert "one or two short sentences" in prompt
    assert "No clinic tools or patient database are connected." in prompt
    assert "Never invent a patient, study, model result, report, appointment, or clinic instruction." in prompt
    assert "Do not provide diagnosis or treatment advice" in prompt
    assert "Do not ask for real patient information, passwords, or API keys." in prompt


def test_manual_language_suite_has_unique_cases_and_documents_its_scope():
    suite = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert suite["schema_version"] == 1
    assert suite["suite_id"] == "response-language-v1"
    assert "not measured model results" in suite["purpose"]
    ids = [case["id"] for case in suite["cases"]]
    assert len(ids) == len(set(ids)) == len(EXPECTED_LANGUAGES)
    assert set(ids) == set(EXPECTED_LANGUAGES)


@pytest.mark.parametrize("case_id,languages", EXPECTED_LANGUAGES.items())
def test_manual_case_has_expected_labels_and_nonempty_inputs(case_id, languages):
    suite = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    case = next(case for case in suite["cases"] if case["id"] == case_id)
    assert [turn["expected_response_language"] for turn in case["turns"]] == languages
    assert all(isinstance(turn["user"], str) and turn["user"].strip() for turn in case["turns"])
    assert case["check"].strip()
