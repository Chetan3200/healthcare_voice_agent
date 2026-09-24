"""Offline lossless tests for first-audio speech chunking."""

import pytest

from healthcare_voice_agent.voice.speech_chunks import split_for_first_audio


@pytest.mark.parametrize("text", [
    "Please bring your previous reports, prescriptions, and insurance card.",
    "कृपया अपनी पुरानी रिपोर्ट, दवाइयाँ, और पहचान पत्र साथ लाएँ।",
    "Kal 10:30 AM par follow-up hai, please reports le aana.",
    "Patient ID 1234567890, BP 120/80, room 42B is confirmed.",
    "prefix " + "x" * 90 + " suffix",
])
def test_split_is_lossless_and_never_whitespace_only(text):
    pieces = split_for_first_audio(text, 32)
    assert "".join(pieces) == text
    assert all(piece.strip() for piece in pieces)
    assert all(piece in text for piece in pieces)


def test_punctuation_boundary_is_preferred_when_available():
    text = "Take the tablet after breakfast. Then call the clinic tomorrow morning."
    pieces = split_for_first_audio(text, 40)
    assert pieces[0] == "Take the tablet after breakfast. "
    assert "".join(pieces) == text


def test_long_word_is_not_split_even_when_it_exceeds_target():
    word = "pneumonoultramicroscopicsilicovolcanoconiosis" * 2
    text = f"Start {word} end"
    pieces = split_for_first_audio(text, 32)
    assert pieces == (text,)  # No punctuation: keep the entire phrase intact.
    assert "".join(pieces) == text


@pytest.mark.parametrize("target", [0, 32, 256])
def test_supported_targets(target):
    assert split_for_first_audio("short text", target) == ("short text",)


@pytest.mark.parametrize("target", [True, False, 1, 31, 257, 1.5, "60"])
def test_invalid_targets_are_rejected(target):
    with pytest.raises(ValueError):
        split_for_first_audio("valid text", target)


def test_whitespace_only_has_no_synthesis_piece():
    assert split_for_first_audio(" \t\n ", 60) == ()


LAKSHADWEEP_RESPONSE = (
    "Yes, Lakshadweep is a group of beautiful islands and a union territory of India, "
    "known for its beaches and marine life."
)


def test_lakshadweep_keeps_marine_life_in_the_same_request():
    assert split_for_first_audio(LAKSHADWEEP_RESPONSE, 60) == (
        "Yes, Lakshadweep is a group of beautiful islands and a union territory of India, ",
        "known for its beaches and marine life.",
    )


def test_long_sentence_without_safe_punctuation_stays_whole():
    text = "The islands are known for their beautiful beaches and colourful coral reefs and abundant marine life."
    assert split_for_first_audio(text, 60) == (text,)


@pytest.mark.parametrize("tail", ["life.", "marine life.", "and the sea."])
def test_short_tail_is_not_its_own_synthesis_request(tail):
    text = "These islands are famous for their sandy beaches and clear coastal waters, " + tail
    assert split_for_first_audio(text, 60) == (text,)


@pytest.mark.parametrize("text", [
    "You should meet the clinician Dr. Sharma at the clinic tomorrow morning.",
    "Your appointment with professor A. Kumar is confirmed for tomorrow morning.",
    "This appointment is arranged in the U.S. territory for tomorrow morning.",
    "Your appointment has been confirmed for 10:30 a.m. tomorrow at the clinic.",
    "Please take the prescribed dose of 2.5 mg after breakfast tomorrow morning.",
    "Please check whether the recorded total is 1, 000 before confirming the booking.",
    "Please check the appointment time written as 10: 30 before confirming tomorrow.",
    'The phrase "beautiful beaches, colourful reefs and marine life" should remain together.',
    "The phrase ‘beautiful beaches, colourful reefs and marine life’ should remain together.",
    "Please bring the records (previous reports, prescriptions and insurance details) tomorrow.",
    "The quoted phrase is \"beautiful beaches, colourful reefs and marine life without an end quote.",
    "Please bring all required documents / previous reports and the identification card tomorrow.",
    "Please bring all required documents - previous reports and the identification card tomorrow.",
])
def test_ambiguous_punctuation_does_not_split_a_phrase(text):
    assert split_for_first_audio(text, 60) == (text,)


@pytest.mark.parametrize("prefix,tail", [
    ("आपकी मुलाकात कल सुबह तय है, ", "कृपया अपनी पिछली रिपोर्ट साथ लेकर आएँ।"),
    ("Your appointment कल सुबह confirmed है, ", "please अपनी reports और पहचान पत्र साथ लाएँ।"),
    ("The first complete thought is ready; ", "the remaining words belong in one request."),
    ("कृपया अपनी पिछली रिपोर्ट साथ लेकर आएँ। ", "आपकी मुलाकात कल सुबह तय की गई है।"),
    ("Please remember the patient's report, ", "and bring the remaining documents tomorrow."),
    ('The guide described "beautiful islands and marine life." ', "Please keep those quoted words together."),
])
def test_supported_boundaries_keep_both_phrases_intact(prefix, tail):
    assert split_for_first_audio(prefix + tail, 60) == (prefix, tail)


def test_only_one_early_cut_even_when_remainder_contains_more_punctuation():
    prefix = "This opening phrase is complete and ready, "
    tail = "the next clause adds more useful information, and the ending mentions marine life."
    assert split_for_first_audio(prefix + tail, 60) == (prefix, tail)


def test_short_greeting_is_not_a_standalone_piece():
    text = "Yes, " + "the islands are famous for their beautiful beaches and rich marine life."
    assert split_for_first_audio(text, 60) == (text,)


def test_exact_whitespace_and_combining_marks_survive_clause_split():
    prefix = "  कृपया अपनी पिछली रिपोर्ट साथ लेकर आएँ,\t\n  "
    tail = "और आगे की जानकारी के लिए हमसे बात करें।  "
    assert split_for_first_audio(prefix + tail, 60) == (prefix, tail)


@pytest.mark.parametrize("target", [32, 40, 60, 128, 256])
def test_no_ordinary_space_cut_for_any_supported_target(target):
    text = "The islands offer beautiful beaches and colourful coral reefs with abundant marine life " * 8 + "."
    assert split_for_first_audio(text, target) == (text,)
