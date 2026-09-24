"""Focused envelope/profile checks; no network, seeding or provider execution."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from healthcare_voice_agent.clinician.flow import functions, initial_node
from healthcare_voice_agent.clinician.records import ClinicalReadError
from healthcare_voice_agent.clinician.runtime import ClinicianRuntime
from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.tools.contracts import ResolvedCaseContext, TOOLS


class Records:
    def read(self, name, args, context):
        if context is None:
            raise ClinicalReadError('IDENTITY_UNRESOLVED')
        if context.case_id != args.case_id:
            raise ClinicalReadError('CASE_CONTEXT_MISMATCH')
        return {'case_id': args.case_id, 'patient_id': 'SYN-P001',
                'as_of': datetime.now(timezone.utc), 'appointments': []}, []


class Guidance:
    def search(self, args):
        return {'query': args.query, 'version_scope': 'web_unversioned', 'provider': 'exa',
                'matches': [], 'as_of': datetime.now(timezone.utc)}, [
                    {'code': 'EXTERNAL_GUIDANCE', 'message': 'Not clinic-approved policy.', 'evidence_ids': []}]


def test_native_tools_and_envelopes():
    async def check():
        runtime = ClinicianRuntime(Records(), Guidance(),
            ResolvedCaseContext(case_id='1042', case_context_version=1), {'case_id': '1042'})
        toolset = functions()
        assert {tool.name for tool in toolset} == set(TOOLS) | {'open_case'}
        assert initial_node(runtime)['name'] == 'clinician'
        assert all(set(tool.required) == set(tool.properties) for tool in toolset)
        args = {'case_id': '1042', 'time_scope': 'all', 'appointment_type': None, 'statuses': None}
        result = await runtime.invoke('get_appointments', args)
        assert result['status'] == 'ok' and result['meta']['case_context_version'] == 1
        assert result['data']['appointments'] == []
        missing = await runtime.invoke('get_appointments', {'case_id': '1042', 'time_scope': 'all'})
        assert missing['error']['code'] == 'INVALID_ARGUMENT'
        mismatch = await runtime.invoke('get_appointments', {**args, 'case_id': '1043'})
        assert mismatch['error']['code'] == 'CASE_CONTEXT_MISMATCH' and mismatch['data'] is None
        runtime.context = None
        result = await runtime.invoke('search_clinic_instructions', {
            'query': 'general wrist fracture guidance', 'top_k': 2, 'document_id': None, 'version': None})
        assert result['status'] == 'ok' and result['contract_version'] == '2.0.0'
        assert result['meta']['case_context_version'] is None
        assert result['warnings'][0]['code'] == 'EXTERNAL_GUIDANCE'
        tool = next(tool for tool in toolset if tool.name == 'get_appointments')
        result, node = await tool.handler(args, SimpleNamespace(state={'session': runtime}))
        assert node is None and result['error']['code'] == 'IDENTITY_UNRESOLVED'
    asyncio.run(check())


def test_profile_secrets_are_not_serialized_and_frontdesk_is_separate():
    env = {'AGENT_MODE': 'clinician', 'EXA_API_KEY': 'private-exa-test',
           'POSTGRES_USER': 'private-db-user-test', 'POSTGRES_PASSWORD': 'private-db-pass-test',
           'POSTGRES_DB': 'synthetic_clinic', 'DB_PORT': '5433'}
    config = load_config(None, environ=env)
    public = json.dumps(config.public_dict())
    assert all(value not in public for value in ('private-exa-test', 'private-db-user-test', 'private-db-pass-test'))
    assert config.clinician.exa_api_key.get_secret_value() in config.redaction_secrets()
    frontdesk = load_config(None, environ={**env, 'AGENT_MODE': 'frontdesk_demo'})
    assert frontdesk.clinician.exa_api_key is None  # No clinical DB/key loading in that profile.
    root = Path(__file__).resolve().parents[2]
    script = (root / 'scripts/voice_clinician.sh').read_text()
    assert 'TTS_VOICE=marin' in script and 'VOICE_PORT:-7863' in script
    assert '--no-sync --no-python-downloads --locked' in script
    assert 'uv sync' not in script and 'alembic upgrade' not in script
    assert 'EXA_API_KEY=' not in script  # Never overwrite the user's key.
