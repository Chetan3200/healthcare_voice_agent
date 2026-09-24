"""Loaded flow version diagnostics without providers, patient context, or raw prompts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from healthcare_voice_agent.voice.flow_fingerprints import flow_prompt_fingerprint
from healthcare_voice_agent.voice.tracing import SessionTrace


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError('No network in fingerprint tests')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)


@pytest.mark.parametrize('mode', ['frontdesk_demo', 'clinician'])
def test_static_role_and_tool_hashes_are_stable_and_contain_no_prompt(mode):
    first = flow_prompt_fingerprint(mode)
    assert first == flow_prompt_fingerprint(mode)
    assert first['agent_mode'] == mode
    for key in ('sha256', 'role_sha256', 'tools_sha256'):
        assert len(first[key]) == 64
        assert all(c in '0123456789abcdef' for c in first[key])
    assert set(first) == {'agent_mode', 'fingerprint_version', 'sha256', 'role_sha256', 'tools_sha256'}


def test_hash_tracks_loaded_role_not_source_file(monkeypatch):
    from healthcare_voice_agent.clinician import flow
    original = flow_prompt_fingerprint('clinician')
    monkeypatch.setattr(flow, '_ROLE', 'private loaded role marker')
    changed = flow_prompt_fingerprint('clinician')
    assert changed['sha256'] != original['sha256']
    assert changed['role_sha256'] != original['role_sha256']
    assert changed['tools_sha256'] == original['tools_sha256']
    assert 'private loaded role marker' not in json.dumps(changed)


def test_native_tool_contract_changes_change_fingerprint(monkeypatch):
    from healthcare_voice_agent.demo import flow
    original = flow_prompt_fingerprint('frontdesk_demo')
    monkeypatch.setattr(flow, 'global_functions', lambda: [SimpleNamespace(
        name='read_only_example', description='private description marker',
        properties={}, required=[], cancel_on_interruption=True, timeout_secs=30)])
    changed = flow_prompt_fingerprint('frontdesk_demo')
    assert changed['tools_sha256'] != original['tools_sha256']
    assert changed['role_sha256'] == original['role_sha256']
    assert 'private description marker' not in json.dumps(changed)


def test_trace_stores_only_hashes_and_no_patient_state(tmp_path, monkeypatch):
    from healthcare_voice_agent.clinician import flow
    monkeypatch.setattr(flow, '_ROLE', 'do not persist this role')
    log = tmp_path / 'events.jsonl'
    trace = SessionTrace(log, session_id='offline')
    trace.emit('flow_prompt_loaded', **flow_prompt_fingerprint('clinician'))
    trace.close()
    contents = log.read_text()
    assert 'do not persist this role' not in contents
    assert json.loads(contents)['event'] == 'flow_prompt_loaded'


def test_generic_conversation_has_no_native_flow_fingerprint():
    assert flow_prompt_fingerprint('conversation') is None


def test_session_records_detailed_fingerprint_as_well_as_base_prompt():
    root = Path(__file__).resolve().parents[2]
    source = (root / 'src/healthcare_voice_agent/voice/session.py').read_text()
    assert 'flow_prompt_fingerprint(config.agent_mode)' in source
    assert 'trace.emit("flow_prompt_loaded", **fingerprint)' in source
