"""Bounded identity/context checks; no database or provider calls."""
import asyncio
import json
import threading
from types import SimpleNamespace

from pipecat.flows import ContextStrategy

from healthcare_voice_agent.clinician import runtime as module
from healthcare_voice_agent.clinician.flow import functions, initial_node
from healthcare_voice_agent.clinician.records import ClinicalReadError
from healthcare_voice_agent.config import load_config
from tests.unit.test_clinician_runtime import Records, Guidance


def identity(case_id):
    return {'case_id': case_id, 'patient_id': 'SYN-' + case_id,
            'patient_display_name': 'Arjun Sharma'}


class Cases(Records):
    def verify_access(self):
        return 2

    def find_cases(self, args):
        if args.patient_name is not None:
            return [identity('1042'), identity('1043')]
        return [identity(args.case_id)] if args.case_id in ('1042', '1043') else []


def runtime(records=None):
    return module.ClinicianRuntime(records or Cases(), Guidance(), None, None)


async def select(session, case_id='1042'):
    return await session.invoke('open_case', {'case_id': case_id, 'patient_name': None})


async def appointments(session, case_id='1042'):
    return await session.invoke('get_appointments', {'case_id': case_id, 'time_scope': 'all',
        'appointment_type': None, 'statuses': None})


def test_starts_unselected_even_if_legacy_environment_supplies_a_case(monkeypatch):
    monkeypatch.setattr(module, 'clinical_records', lambda settings: Cases())
    async def check():
        config = load_config(None, environ={'AGENT_MODE': 'clinician', 'CLINICIAN_CASE_ID': '1042'})
        session = await module.open_clinician_runtime(config)
        assert session.context is None and session.summary is None
        assert (await appointments(session))['error']['code'] == 'IDENTITY_UNRESOLVED'
        assert '1042' not in json.dumps(initial_node(session)['task_messages'])
    asyncio.run(check())


def test_exact_selection_correction_and_old_case_rejection():
    async def check():
        session = runtime()
        first = await select(session)
        assert first['data']['selected_case']['case_id'] == '1042'
        assert 'date_of_birth' not in first['data']['selected_case']
        assert 'date_of_birth' not in session.summary
        assert first['meta']['case_context_version'] == 1
        assert (await appointments(session))['status'] == 'ok'
        second = await select(session, '1043')
        assert second['meta']['case_context_version'] == 2
        assert session.context.case_id == '1043'
        assert (await appointments(session))['error']['code'] == 'CASE_CONTEXT_MISMATCH'
        assert (await appointments(session, '1043'))['status'] == 'ok'
    asyncio.run(check())


def test_name_ambiguity_returns_identity_only_and_clears_old_case():
    async def check():
        session = runtime()
        await select(session)
        response = await session.invoke('open_case', {'case_id': None, 'patient_name': 'Arjun Sharma'})
        assert response['data']['state'] == 'selection_required'
        assert len(response['data']['candidates']) == 2
        assert session.context is None and session.summary is None
        assert set(response['data']['candidates'][0]) == {
            'case_id', 'patient_id', 'patient_display_name'}
        assert (await appointments(session))['error']['code'] == 'IDENTITY_UNRESOLVED'
    asyncio.run(check())


def test_bad_switches_never_fall_back_to_old_patient():
    async def check():
        session = runtime()
        for arguments, code in [
            ({'case_id': '9999', 'patient_name': None}, 'CASE_NOT_FOUND'),
            ({'case_id': '1042'}, 'INVALID_ARGUMENT'),
            ({'case_id': '1042', 'patient_name': 'Arjun'}, 'INVALID_ARGUMENT'),
            ({'case_id': '1042', 'patient_name': 'None'}, 'INVALID_ARGUMENT'),
            ({'case_id': None, 'patient_name': None}, 'INVALID_ARGUMENT'),
        ]:
            await select(session)
            result = await session.invoke('open_case', arguments)
            assert result['error']['code'] == code
            assert session.context is None
            assert (await appointments(session))['error']['code'] == 'IDENTITY_UNRESOLVED'
    asyncio.run(check())


def test_pending_old_read_is_discarded_after_case_correction():
    started, release = threading.Event(), threading.Event()
    class Slow(Cases):
        def read(self, name, args, context):
            started.set()
            assert release.wait(2)
            return super().read(name, args, context)
    async def check():
        session = runtime(Slow())
        await select(session)
        pending = asyncio.create_task(appointments(session))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            await select(session, '1043')
        finally:
            release.set()
        response = await pending
        assert response['data'] is None and response['error']['code'] == 'CASE_CONTEXT_MISMATCH'
        assert response['meta']['case_context_version'] == 1
    asyncio.run(check())


def test_late_case_lookup_cannot_restore_obsolete_selection():
    started, release = threading.Event(), threading.Event()
    class Slow(Cases):
        def find_cases(self, args):
            if args.case_id == '1042':
                started.set()
                assert release.wait(2)
            return super().find_cases(args)
    async def check():
        session = runtime(Slow())
        pending = asyncio.create_task(select(session))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            assert (await select(session, '1043'))['status'] == 'ok'
        finally:
            release.set()
        response = await pending
        assert response['error']['code'] == 'CASE_CONTEXT_MISMATCH'
        assert session.context.case_id == '1043' and session.context.case_context_version == 2
    asyncio.run(check())


def test_case_selection_uses_native_same_node_reset_without_old_evidence():
    async def check():
        session = runtime()
        history = [
            {'role': 'user', 'content': 'Open case 1042.'},
            {'role': 'assistant', 'content': 'Old patient clinical evidence must not carry across.'},
            {'role': 'user', 'content': 'Open case 1043 and show the reviewed report.'},
        ]
        manager = SimpleNamespace(state={'session': session}, get_current_context=lambda: history)
        tool = next(tool for tool in functions() if tool.name == 'open_case')
        response, node = await tool.handler({'case_id': '1043', 'patient_name': None}, manager)
        assert response['status'] == 'ok' and node['name'] == 'clinician'
        assert node['context_strategy'].strategy == ContextStrategy.RESET
        assert node['task_messages'][-1] == {'role': 'user',
            'content': [{'type': 'text', 'text': history[-1]['content']}]}
        assert 'Old patient clinical evidence' not in json.dumps(node['task_messages'])
        assert '1042' not in json.dumps(node['task_messages'])
    asyncio.run(check())


def test_terminal_failed_selection_resets_without_replaying_imperative_request():
    async def check():
        session = runtime()
        await select(session)
        request = {'role': 'user', 'content': 'Open missing9999 and show the reviewed report.'}
        history = [{'role': 'assistant', 'content': 'Old case evidence.'}, request]
        manager = SimpleNamespace(state={'session': session}, get_current_context=lambda: history)
        tool = next(tool for tool in functions() if tool.name == 'open_case')
        response, node = await tool.handler(
            {'case_id': 'missing9999', 'patient_name': None}, manager)
        assert response['error']['code'] == 'CASE_NOT_FOUND'
        assert session.context is None and session.summary is None
        assert node['context_strategy'].strategy == ContextStrategy.RESET
        assert len(node['task_messages']) == 1
        message = node['task_messages'][0]['content']
        assert 'ask for a corrected exact case id or patient name' in message.lower()
        assert 'For a temporary backend failure' in message
        assert 'do not claim the identifier was wrong' in message
        assert 'Do not claim or offer candidate IDs' in message
        assert 'missing9999 and show' not in message
        assert 'Old case evidence' not in json.dumps(node['task_messages'])
    asyncio.run(check())


def test_ambiguous_selection_does_not_replay_lookup_and_corrected_next_turn_is_preserved():
    async def check():
        session = runtime()
        tool = next(tool for tool in functions() if tool.name == 'open_case')
        ambiguous_request = {'role': 'user', 'content': 'Open Arjun Sharma and show the study.'}
        manager = SimpleNamespace(
            state={'session': session}, get_current_context=lambda: [ambiguous_request])
        response, node = await tool.handler(
            {'case_id': None, 'patient_name': 'Arjun Sharma'}, manager)
        assert response['data']['state'] == 'selection_required'
        assert len(node['task_messages']) == 1
        message = node['task_messages'][0]['content']
        assert 'offer only the returned candidate case IDs' in message
        assert 'Do not rerun the name lookup' in message
        assert ambiguous_request['content'] not in message

        corrected = {'role': 'user', 'content': 'Use case 1043 and show the study.'}
        manager.get_current_context = lambda: node['task_messages'] + [corrected]
        response, node = await tool.handler(
            {'case_id': '1043', 'patient_name': None}, manager)
        assert response['data']['state'] == 'selected'
        assert node['task_messages'][-1] == {
            'role': 'user', 'content': [{'type': 'text', 'text': corrected['content']}]}
        assert ambiguous_request['content'] not in json.dumps(node['task_messages'])
    asyncio.run(check())


def test_native_flow_manager_emits_context_replacement_on_same_node():
    from unittest.mock import AsyncMock, Mock
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.flows import FlowManager
    from pipecat.frames.frames import LLMMessagesUpdateFrame
    async def check():
        session = runtime()
        worker = Mock(spec=PipelineWorker)
        worker.queue_frames = AsyncMock()
        manager = FlowManager(llm=SimpleNamespace(), context_aggregator=SimpleNamespace(),
                              worker=worker, global_functions=functions())
        manager.state['session'] = session
        await manager.initialize(initial_node(session))
        result = await select(session)
        await manager.set_node_from_config(initial_node(session, selection_result=result,
            request={'role': 'user', 'content': 'Open case 1042. Literal {{untrusted}} text.'}))
        frames = [frame for call in worker.queue_frames.await_args_list for frame in call.args[0]]
        updates = [frame for frame in frames if isinstance(frame, LLMMessagesUpdateFrame)]
        assert len(updates) == 2
        assert 'null' in updates[0].messages[0]['content']
        assert '1042' in updates[1].messages[0]['content']
        assert updates[1].messages[-1]['role'] == 'user'
        assert '{{untrusted}}' in updates[1].messages[-1]['content'][0]['text']
        assert manager.current_node == 'clinician'
    asyncio.run(check())


def test_case_identity_schema_rejects_birth_dates():
    import pytest
    from pydantic import ValidationError
    from healthcare_voice_agent.clinician.case_selection import CaseIdentity, OPEN_CASE
    assert 'date_of_birth' not in CaseIdentity.model_json_schema()['properties']
    assert 'date_of_birth' not in json.dumps(OPEN_CASE.export())
    with pytest.raises(ValidationError):
        CaseIdentity.model_validate({**identity('1042'), 'date_of_birth': '1996-09-21'})


def test_prompt_and_tool_descriptions_keep_privacy_and_dependency_instructions():
    # Static wiring checks only, not evidence of live model compliance or pronunciation.
    node = initial_node(runtime())
    prompt = node['role_message']
    assert 'Do not volunteer, enumerate or repeat patients\' birth dates' in prompt
    assert '21st September 1996' in prompt
    assert 'independent reads may run in parallel' in prompt
    assert 'status=ok alone does NOT mean a case is selected' in prompt
    tools = {tool.name: tool for tool in functions()}
    assert 'Do not disclose or enumerate birth dates' in tools['open_case'].description
    for name in ('get_study', 'get_model_result', 'get_reviewed_report', 'get_appointments'):
        assert 'status=ok AND data.state=selected' in tools[name].description
        assert 'Never batch with open_case' in tools[name].description
    assert 'Requires a selected case' not in tools['search_clinic_instructions'].description


def test_tool_sequence_examples_use_valid_arguments_and_remain_in_the_prompt():
    # Validate static examples, not live LLM sequencing or error interpretation.
    import re
    from healthcare_voice_agent.clinician.case_selection import OPEN_CASE
    from healthcare_voice_agent.tools.contracts import TOOLS
    prompt = initial_node(runtime())['role_message']
    calls = re.findall(r'^(open_case|get_study|get_appointments): (\{[^\n]+\})$',
                       prompt, flags=re.MULTILINE)
    assert [name for name, _ in calls] == [
        'open_case', 'get_study', 'get_study', 'get_appointments']
    contracts = {'open_case': OPEN_CASE, **TOOLS}
    for name, arguments in calls:
        contracts[name].validate_arguments_json(arguments)
    assert json.loads(calls[0][1]) == {'case_id': '<corrected_case_id>', 'patient_name': None}
    assert json.loads(calls[1][1]) == {'case_id': '<corrected_case_id>', 'study_id': None}
    assert {json.loads(arguments)['case_id'] for _, arguments in calls} == {
        '<corrected_case_id>', '<selected_case_id>'}
    assert '1042' not in prompt and '1043' not in prompt
    assert 'symbolic placeholders, never\nliteral tool arguments' in prompt
    assert 'Round 1, ONLY:' in prompt and 'Round 2:' in prompt
    assert 'Same round, in parallel:' in prompt
    assert 'This is NOT evidence that open_case failed or the case is missing.' in prompt
    assert 'Only say a case was not found when open_case for that ID returns CASE_NOT_FOUND.' in prompt


def test_prompt_and_tool_descriptions_preserve_missing_fields_and_request_scope():
    # Static prompt/schema wiring only; live model compliance still needs testing.
    prompt = initial_node(runtime())['role_message']
    wording = ' '.join(prompt.split())
    assert 'Examination date/time comes ONLY from performed_at.' in wording
    assert 'source.updated_at' in wording and 'meta.retrieved_at' in wording
    assert 'neither is an examination time' in wording
    assert 'study-only request, do not fetch or volunteer appointment information' in wording
    assert 'potentially meaningful ID prefix or guess missing digits' in wording
    assert '"OpenCAS" can mean "open case", not an ID prefix' in wording
    example = next(line for line in prompt.splitlines()
                   if line.startswith('{"acquisition_status":'))
    fields = json.loads(example)
    assert fields == {
        'acquisition_status': 'completed', 'performed_at': None,
        'laterality': 'unknown', 'image_ref': None,
        'source': {'updated_at': '<record_update_timestamp>'},
    }
    tools = {tool.name: tool for tool in functions()}
    study_description = ' '.join(tools['get_study'].description.split())
    assert 'Never substitute either for a missing performed_at' in study_description
    assert 'image_ref=null means no image-viewing reference is available' in study_description
    assert 'Unknown laterality must remain unknown' in study_description
    appointment_description = ' '.join(tools['get_appointments'].description.split())
    assert 'not as a routine companion to a study or report lookup' in appointment_description


def test_nullable_tool_fields_explicitly_describe_json_null():
    from healthcare_voice_agent.clinician.case_selection import OPEN_CASE
    from healthcare_voice_agent.tools.contracts import TOOLS

    exposed_tools = {tool.name: tool for tool in functions()}
    nullable_fields = []
    for name, contract in {'open_case': OPEN_CASE, **TOOLS}.items():
        schema = contract.arguments_model.model_json_schema()
        for field, field_schema in schema['properties'].items():
            if not any(option.get('type') == 'null'
                       for option in field_schema.get('anyOf', [])):
                continue
            nullable_fields.append((name, field))
            assert field in schema['required']
            description = exposed_tools[name].properties[field]['description']
            assert 'JSON null' in description
            assert 'Never send the strings "None" or "null"' in description
    assert len(nullable_fields) == 9


def test_null_like_record_strings_are_not_silently_repaired():
    from healthcare_voice_agent.tools.contracts import GetStudyArgs

    assert GetStudyArgs(case_id='1042', study_id=None).study_id is None
    for value in ('None', 'null'):
        assert GetStudyArgs(case_id='1042', study_id=value).study_id == value


def test_model_dependency_history_and_score_prompt_rules_are_wired():
    prompt = ' '.join(initial_node(runtime())['role_message'].split())
    assert 'Never send a null, missing or placeholder study_id' in prompt
    assert 'Use an already retrieved study ID instead of asking the caller' in prompt
    assert 'Do not repeat a successful lookup with the same arguments' in prompt
    assert 'Do not emit duplicate calls with identical arguments in the same batch' in prompt
    assert 'Never construct IDs by changing suffixes' in prompt
    assert 'One failed ID lookup cannot prove there are no historical results' in prompt
    assert 'choose the first historical ID by list order' in prompt
    assert 'current_model_result reports the designated current record' in prompt
    assert 'current_model_result is pending, failed or absent' in prompt
    assert 'low dislocation presence score is not confidence in' in prompt
    tools = {tool.name: ' '.join(tool.description.split()) for tool in functions()}
    assert 'get_study.model_results' in tools['get_model_result']
    assert 'Never select the first historical ID' in tools['get_model_result']
    assert 'Do not batch with a lookup needed to discover that ID' in tools['get_model_result']
    assert 'Do not batch with a lookup needed to discover that ID' in tools['get_reviewed_report']
