"""Focused fixture-backed mapping tests; this fake does not verify live PostgreSQL SQL."""

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from healthcare_voice_agent.clinician.records import ClinicalReadError, ClinicalRecords
from healthcare_voice_agent.tools.contracts import (
    GetAppointmentsArgs, GetModelResultArgs, GetStudyArgs, ResolvedCaseContext,
)

NOW = datetime(2026, 9, 17, 9, tzinfo=timezone.utc)
CASE = {"case_id": "1042", "patient_id": "SYN-P001", "patient_display_name": "Arjun Sharma", "description": "synthetic"}
STUDY = {"study_id": "ST-1042-01", "case_id": "1042", "performed_at": NOW, "modality": "XR", "body_part": "wrist", "laterality": "left", "views": ["PA"], "acquisition_status": "completed", "image_ref": "fixture://image", "record_version": 3, "updated_at": NOW}


class Cursor:
    def __init__(self, rows):
        self.rows, self.current, self.calls = rows, [], []

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        if "FROM clinic.authorized_cases AS ac" in sql:
            self.current = [CASE] if params == ("SYN-USER-CLIN", "1042") else []
        elif "FROM clinic.studies" in sql:
            self.current = self.rows
        else:
            self.current = []

    def fetchone(self): return self.current[0] if self.current else None
    def fetchall(self): return list(self.current)


class Connection:
    def __init__(self, rows): self.cursor_instance = Cursor(rows)
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def cursor(self): return self.cursor_instance


def repository(studies):
    connections = []
    @contextmanager
    def connect():
        connection = Connection(studies)
        connections.append(connection)
        yield connection
    return ClinicalRecords(connect, "SYN-USER-CLIN"), connections


def context(case_id="1042"):
    return ResolvedCaseContext(case_id=case_id, case_context_version=1)


def test_confirm_case_returns_only_authorized_synthetic_summary():
    records, connections = repository([STUDY])
    assert records.confirm_case("1042") == CASE
    sql, params = connections[0].cursor_instance.calls[0]
    assert "clinic.authorized_cases" in sql and "p.is_synthetic" in sql
    assert params == ("SYN-USER-CLIN", "1042")


def test_study_read_maps_versioned_evidence():
    records, _ = repository([STUDY])
    data, warnings = records.read("get_study", GetStudyArgs(case_id="1042", study_id="ST-1042-01"), context())
    assert warnings == []
    assert data["patient_id"] == "SYN-P001"
    assert data["source"] == {"evidence_id": "study:ST-1042-01:v3:2026-09-17T09:00:00+00:00", "source_type": "study", "record_id": "ST-1042-01", "version": "3", "updated_at": NOW}


def test_missing_or_stale_context_does_not_connect():
    records, connections = repository([STUDY])
    args = GetStudyArgs(case_id="1042", study_id="ST-1042-01")
    with pytest.raises(ClinicalReadError, match="IDENTITY_UNRESOLVED") as error:
        records.read("get_study", args, None)
    assert error.value.code == "IDENTITY_UNRESOLVED"
    with pytest.raises(ClinicalReadError, match="CASE_CONTEXT_MISMATCH") as error:
        records.read("get_study", args, context("1043"))
    assert error.value.code == "CASE_CONTEXT_MISMATCH"
    assert connections == []


def test_unselected_multiple_studies_returns_same_case_candidates():
    second = {**STUDY, "study_id": "ST-1042-02", "performed_at": datetime(2026, 9, 18, tzinfo=timezone.utc)}
    records, connections = repository([STUDY, second])
    with pytest.raises(ClinicalReadError) as error:
        records.read("get_study", GetStudyArgs(case_id="1042", study_id=None), context())
    assert error.value.code == "AMBIGUOUS_RECORD"
    assert [item["record_id"] for item in error.value.candidates] == ["ST-1042-01", "ST-1042-02"]
    assert not any('FROM clinic.model_results' in sql for connection in connections
                   for sql, _ in connection.cursor_instance.calls)


def test_appointments_convert_instants_to_the_recorded_timezone():
    appointment = {
        "appointment_id": "AP-1042-01", "appointment_type": "fracture_follow_up",
        "starts_at": NOW, "ends_at": None, "timezone": "Asia/Kolkata",
        "status": "scheduled", "clinician_id": None, "clinician_display_name": None,
        "location": None, "notes": None, "replaces_appointment_id": None,
        "record_version": 2, "updated_at": NOW,
    }

    class AppointmentCursor(Cursor):
        def execute(self, sql, params=()):
            self.calls.append((sql, params))
            if "FROM clinic.authorized_cases AS ac" in sql:
                self.current = [CASE]
            elif "CURRENT_TIMESTAMP" in sql:
                self.current = [{"as_of": NOW}]
            elif "FROM clinic.appointments" in sql:
                self.current = [appointment]
            else:
                self.current = []

    @contextmanager
    def connect():
        connection = Connection([])
        connection.cursor_instance = AppointmentCursor([])
        yield connection

    records = ClinicalRecords(connect, "SYN-USER-CLIN")
    data, warnings = records.read(
        "get_appointments",
        GetAppointmentsArgs(case_id="1042", time_scope="all", appointment_type=None, statuses=None),
        context(),
    )
    assert warnings == []
    assert data["appointments"][0]["starts_at"].isoformat() == "2026-09-17T14:30:00+05:30"


@pytest.mark.parametrize('identifier', [{'case_id': '1042', 'patient_name': None},
                                      {'case_id': None, 'patient_name': 'Arjun Sharma'}])
def test_case_lookup_does_not_query_birth_dates(identifier):
    from healthcare_voice_agent.clinician.case_selection import OpenCaseArgs
    records, connections = repository([])
    records.find_cases(OpenCaseArgs(**identifier))
    sql, params = connections[0].cursor_instance.calls[0]
    projection = sql.split('FROM')[0].lower()
    assert 'date_of_birth' not in projection and '*' not in projection
    assert 'clinic.authorized_cases' in sql and 'p.is_synthetic' in sql
    assert params == ('SYN-USER-CLIN', identifier['case_id'] or identifier['patient_name'])


def test_study_exposes_current_status_separately_from_completed_history():
    from healthcare_voice_agent.tools.contracts import StudyData
    current = {'model_result_id': 'MR-1052-01-V2', 'model_version': '2.0.0',
               'record_version': 2, 'inference_status': 'pending',
               'record_status': 'active', 'is_current': True}
    references = [
        {'model_result_id': 'MR-1052-01-V1', 'model_version': '1.0.0',
         'record_version': 1, 'is_current': False},
    ]
    connections = []
    class ReferenceCursor(Cursor):
        def execute(self, sql, params=()):
            if 'FROM clinic.model_results' in sql:
                self.calls.append((sql, params))
                self.current = [current] if "mr.record_status = 'active'" in sql else references
            else:
                super().execute(sql, params)
    @contextmanager
    def connect():
        connection = Connection([STUDY])
        connection.cursor_instance = ReferenceCursor([STUDY])
        connections.append(connection)
        yield connection
    records = ClinicalRecords(connect, 'SYN-USER-CLIN')
    data, _ = records.read('get_study', GetStudyArgs(case_id='1042', study_id=None), context())
    assert data['current_model_result'] == current
    assert data['model_results'] == references
    parsed = StudyData.model_validate(data)
    assert parsed.current_model_result.inference_status == 'pending'
    assert len(parsed.model_results) == 1
    current_sql, current_params = connections[-1].cursor_instance.calls[-2]
    history_sql, history_params = connections[-1].cursor_instance.calls[-1]
    assert current_params == history_params == ('SYN-USER-CLIN', '1042', 'ST-1042-01')
    assert 'ac.user_id = %s AND ac.case_id = %s AND s.study_id = %s' in current_sql
    assert 'p.is_synthetic' in current_sql and "mr.record_status = 'active'" in current_sql
    assert 'mr.is_current' in current_sql
    assert "mr.inference_status = 'completed'" in history_sql
    assert "mr.record_status IN ('active', 'superseded')" in history_sql
    assert '*' not in current_sql.split('FROM')[0] and '*' not in history_sql.split('FROM')[0]
    assert set(references[0]) == {'model_result_id', 'model_version', 'record_version', 'is_current'}


def test_empty_model_reference_list_is_distinct_from_older_unenumerated_response():
    from healthcare_voice_agent.tools.contracts import StudyData
    records, _ = repository([STUDY])
    data, _ = records.read('get_study', GetStudyArgs(case_id='1042', study_id=None), context())
    parsed = StudyData.model_validate(data)
    assert parsed.current_model_result is None
    assert parsed.model_results == []
    data.pop('model_results')
    data.pop('current_model_result')
    older = StudyData.model_validate(data)
    assert older.model_results is None and older.current_model_result is None


@pytest.mark.parametrize('inference_status', ['pending', 'failed', 'completed'])
def test_current_model_status_supports_available_and_unavailable_lifecycle_states(inference_status):
    from healthcare_voice_agent.tools.contracts import StudyData
    current = {'model_result_id': 'MR-1042-01-CURRENT', 'model_version': '2.0.0',
               'record_version': 2, 'inference_status': inference_status,
               'record_status': 'active', 'is_current': True}
    class StatusCursor(Cursor):
        def execute(self, sql, params=()):
            if 'FROM clinic.model_results' in sql:
                self.calls.append((sql, params))
                self.current = [current] if "mr.record_status = 'active'" in sql else []
            else:
                super().execute(sql, params)
    @contextmanager
    def connect():
        connection = Connection([STUDY])
        connection.cursor_instance = StatusCursor([STUDY])
        yield connection
    records = ClinicalRecords(connect, 'SYN-USER-CLIN')
    data, _ = records.read('get_study', GetStudyArgs(case_id='1042', study_id=None), context())
    parsed = StudyData.model_validate(data)
    assert parsed.current_model_result.inference_status == inference_status
    assert parsed.model_results == []


def test_multiple_designated_current_model_results_are_a_conflict():
    current = {'model_result_id': 'MR-1042-01-V2', 'model_version': '2.0.0',
               'record_version': 2, 'inference_status': 'pending',
               'record_status': 'active', 'is_current': True}
    class ConflictCursor(Cursor):
        def execute(self, sql, params=()):
            if 'FROM clinic.model_results' in sql:
                self.calls.append((sql, params))
                self.current = [current, {**current, 'model_result_id': 'MR-1042-01-V3'}]
            else:
                super().execute(sql, params)
    @contextmanager
    def connect():
        connection = Connection([STUDY])
        connection.cursor_instance = ConflictCursor([STUDY])
        yield connection
    records = ClinicalRecords(connect, 'SYN-USER-CLIN')
    with pytest.raises(ClinicalReadError, match='RECORD_CONFLICT') as error:
        records.read('get_study', GetStudyArgs(case_id='1042', study_id=None), context())
    assert error.value.code == 'RECORD_CONFLICT'


def test_case_1052_pending_current_is_visible_but_default_model_read_does_not_fall_back():
    case_1052 = {**CASE, 'case_id': '1052', 'patient_id': 'SYN-P052'}
    study_1052 = {**STUDY, 'study_id': 'ST-1052-01', 'case_id': '1052'}
    pending_v2 = {
        'model_result_id': 'MR-1052-01-V2', 'study_id': 'ST-1052-01',
        'model_name': 'synthetic-fixture-model', 'model_version': '2.0.0',
        'inference_status': 'pending', 'generated_at': None,
        'record_status': 'active', 'is_current': True, 'summary': None,
        'score_description': None, 'limitations': [], 'record_version': 2,
        'updated_at': NOW,
    }
    completed_v1 = {
        'model_result_id': 'MR-1052-01-V1', 'study_id': 'ST-1052-01',
        'model_name': 'synthetic-fixture-model', 'model_version': '1.0.0',
        'inference_status': 'completed', 'generated_at': NOW,
        'record_status': 'superseded', 'is_current': False,
        'summary': 'Historical synthetic result.', 'score_description': 'Stored score.',
        'limitations': [], 'record_version': 1, 'updated_at': NOW,
    }

    class Case1052Cursor(Cursor):
        def execute(self, sql, params=()):
            self.calls.append((sql, params))
            if 'FROM clinic.authorized_cases AS ac' in sql and 'JOIN clinic.cases AS c' in sql:
                self.current = [case_1052] if params == ('SYN-USER-CLIN', '1052') else []
            elif 'FROM clinic.studies AS s' in sql:
                self.current = [study_1052]
            elif 'FROM clinic.model_results AS mr' in sql:
                if 'mr.model_result_id = %s' in sql:
                    self.current = [completed_v1] if params[-1] == 'MR-1052-01-V1' else []
                elif "mr.inference_status = 'completed'" in sql and "mr.record_status = 'active'" in sql:
                    self.current = []
                elif "mr.record_status = 'active'" in sql:
                    self.current = [{key: pending_v2[key] for key in (
                        'model_result_id', 'model_version', 'record_version',
                        'inference_status', 'record_status', 'is_current')}]
                else:
                    self.current = [{key: completed_v1[key] for key in (
                        'model_result_id', 'model_version', 'record_version', 'is_current')}]
            elif 'FROM clinic.model_findings' in sql:
                self.current = []
            else:
                self.current = []

    @contextmanager
    def connect():
        connection = Connection([])
        connection.cursor_instance = Case1052Cursor([])
        yield connection

    records = ClinicalRecords(connect, 'SYN-USER-CLIN')
    selected = ResolvedCaseContext(case_id='1052', case_context_version=1)
    study_data, _ = records.read(
        'get_study', GetStudyArgs(case_id='1052', study_id='ST-1052-01'), selected)
    assert study_data['current_model_result']['model_result_id'] == 'MR-1052-01-V2'
    assert study_data['current_model_result']['inference_status'] == 'pending'
    assert [item['model_result_id'] for item in study_data['model_results']] == ['MR-1052-01-V1']

    with pytest.raises(ClinicalReadError, match='RESULT_NOT_AVAILABLE') as error:
        records.read('get_model_result', GetModelResultArgs(
            case_id='1052', study_id='ST-1052-01', model_result_id=None), selected)
    assert error.value.code == 'RESULT_NOT_AVAILABLE'

    historical, warnings = records.read('get_model_result', GetModelResultArgs(
        case_id='1052', study_id='ST-1052-01', model_result_id='MR-1052-01-V1'), selected)
    assert historical['model_result_id'] == 'MR-1052-01-V1'
    assert warnings[0]['code'] == 'HISTORICAL_RECORD'
