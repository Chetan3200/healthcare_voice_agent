"""Offline static assertions for clinician prompt safety requirements."""
from types import SimpleNamespace

from healthcare_voice_agent.clinician.flow import initial_node


def prompt():
    return ' '.join(initial_node(SimpleNamespace(summary=None))['role_message'].split())


def test_case_id_normalization_preserves_latest_correction_exactly():
    text = prompt()
    assert "normalized number exactly matches the caller's latest stated or corrected ID" in text
    assert 'add or drop no digits' in text
    assert 'Clarify only when that ID is genuinely uncertain' in text


def test_appointment_only_reads_and_fields_are_constrained():
    text = prompt()
    assert 'appointment-only queries, call get_appointments and selection prerequisites only' in text
    assert 'do not retrieve studies, model results, or reports' in text
    assert 'records cannot determine which booking is correct' in text
    assert 'refer the conflict to scheduling staff' in text
    assert 'A caller preference is not proof of the intended booking' in text
    assert 'including a missing end time as unavailable' in text
    assert 'Keep starts_at and ends_at distinct; never substitute one for the other' in text
    assert 'state its source timezone explicitly; never present UTC as local time' in text


def test_read_only_status_requests_do_not_offer_writes():
    text = prompt()
    assert 'never offer or attempt booking, cancellation, or rescheduling' in text
    assert 'For an appointment status request, answer the status and stop' in text
    assert 'Refer explicit write requests to the front desk' in text


def test_guidance_is_spoken_conversationally_without_report_formatting():
    text = prompt()
    assert 'brief conversation, not a written report' in text
    assert 'short connected sentences' in text
    assert 'Do not omit important safety warnings or source differences' in text
    assert 'Markdown links, URLs, numbered lists, bullet symbols or bracketed source labels' in text
    assert 'Name publishers naturally once' in text
    assert 'keep different patient groups, follow-up intervals and disagreements separate' in text
    assert "Honour the caller's requested source count" in text
    assert 'without restarting the full list of sources' in text


def test_guidance_examples_are_not_evidence_and_quotations_stay_verbatim():
    text = prompt()
    assert 'Speech-style examples ONLY, not retrieved evidence or advice for the active case' in text
    assert 'never these example facts or names without real returned support' in text
    assert 'use only exact words from returned passages' in text
    assert 'Never pass a paraphrase off as a quotation' in text
    assert 'In plain English' in text
    assert "not your clinic's own policy" in text


def test_source_followup_does_not_trigger_case_selection_or_invent_url_limits():
    text = prompt()
    assert 'publisher or leaflet is still guidance, not a patient lookup' in text
    assert 'do not call open_case for it' in text
    assert 'if several pages match, ask which one' in text
    assert 'use its returned URL as document_id' in text
    assert 'generic clinical query and version=null' in text
    assert 'Do not claim URL-scoped search is unsupported' in text
    assert 'extracted passages are the entire page' in text


def test_garbled_clinical_term_gets_only_narrow_clarification():
    text = prompt()
    assert 'Did you mean wrist cast?' in text
    assert 'transcript says "risk caste"' in text
    assert 'Do not add consent or confirmation steps for a clear request' in text
