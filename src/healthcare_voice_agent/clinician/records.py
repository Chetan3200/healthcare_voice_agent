"""Read-only clinical record retrieval backed by parameterized PostgreSQL queries."""

from __future__ import annotations

from typing import Any, Callable
from zoneinfo import ZoneInfo

from healthcare_voice_agent.tools.contracts import ResolvedCaseContext


class ClinicalReadError(Exception):
    """Safe, caller-facing clinical retrieval error."""

    def __init__(self, code: str, candidates: list[dict[str, Any]] | None = None) -> None:
        self.code = code
        self.candidates = candidates or []
        super().__init__(code)


class ClinicalRecords:
    """Small read-only repository; connection setup is owned by the caller."""

    def __init__(self, connect: Callable[[], Any], user_id: str) -> None:
        self._connect = connect
        self._user_id = user_id

    def verify_access(self) -> int:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT count(*) AS case_count FROM clinic.authorized_cases AS ac
                   JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                   WHERE ac.user_id = %s AND p.is_synthetic""", (self._user_id,))
            count = cursor.fetchone()["case_count"]
        if not count:
            raise ClinicalReadError("CASE_NOT_FOUND")
        return count

    def find_cases(self, args) -> list[dict[str, Any]]:
        # Literal name substring lookup, not fuzzy matching or caller-language parsing.
        condition = "ac.case_id = %s" if args.case_id is not None else "strpos(lower(p.display_name), lower(%s)) > 0"
        identifier = args.case_id if args.case_id is not None else args.patient_name.strip()
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT ac.case_id, ac.patient_id, p.display_name AS patient_display_name
                   FROM clinic.authorized_cases AS ac
                   JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                   WHERE ac.user_id = %s AND p.is_synthetic AND """ + condition +
                " ORDER BY ac.case_id LIMIT 51", (self._user_id, identifier))
            return [dict(row) for row in cursor.fetchall()]

    def confirm_case(self, case_id: str) -> dict[str, Any]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.case_id, c.patient_id, p.display_name AS patient_display_name,
                       c.description
                FROM clinic.authorized_cases AS ac
                JOIN clinic.cases AS c ON c.case_id = ac.case_id
                JOIN clinic.patients AS p ON p.patient_id = c.patient_id
                WHERE ac.user_id = %s AND ac.case_id = %s AND p.is_synthetic
                """,
                (self._user_id, case_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise ClinicalReadError("CASE_NOT_FOUND")
        return dict(row)

    def read(
        self, tool_name: str, args: Any, context: ResolvedCaseContext | None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        case_id = getattr(args, "case_id", None)
        if context is None:
            raise ClinicalReadError("IDENTITY_UNRESOLVED")
        if context.case_id != case_id:
            raise ClinicalReadError("CASE_CONTEXT_MISMATCH")
        # Authenticate and prove the exact case before accepting any child ID.
        case = self.confirm_case(case_id)
        if tool_name == "get_study":
            return self._get_study(args, case)
        if tool_name == "get_model_result":
            return self._get_model_result(args, case)
        if tool_name == "get_reviewed_report":
            return self._get_reviewed_report(args, case)
        if tool_name == "get_appointments":
            return self._get_appointments(args, case)
        raise ClinicalReadError("INVALID_ARGUMENT")

    def _get_study(self, args: Any, case: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with self._connect() as connection, connection.cursor() as cursor:
            if args.study_id is not None:
                cursor.execute(
                    """SELECT s.* FROM clinic.studies AS s
                       JOIN clinic.authorized_cases AS ac ON ac.case_id = s.case_id
                       JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                       WHERE ac.user_id = %s AND ac.case_id = %s AND s.study_id = %s
                         AND p.is_synthetic""",
                    (self._user_id, case["case_id"], args.study_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ClinicalReadError("RECORD_NOT_FOUND")
            else:
                cursor.execute(
                    """SELECT s.* FROM clinic.studies AS s
                       JOIN clinic.authorized_cases AS ac ON ac.case_id = s.case_id
                       JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                       WHERE ac.user_id = %s AND ac.case_id = %s AND p.is_synthetic
                       ORDER BY s.performed_at NULLS LAST, s.study_id""",
                    (self._user_id, case["case_id"]),
                )
                rows = cursor.fetchall()
                if not rows:
                    raise ClinicalReadError("RECORD_NOT_FOUND")
                if len(rows) != 1:
                    raise ClinicalReadError("AMBIGUOUS_RECORD", [self._study_candidate(x) for x in rows])
                row = rows[0]
            # Current lifecycle metadata and historical discovery share the study's
            # authorization and read connection, but remain separate concepts.
            scope = (self._user_id, case["case_id"], row["study_id"])
            cursor.execute(
                """SELECT mr.model_result_id, mr.model_version, mr.record_version,
                          mr.inference_status, mr.record_status, mr.is_current
                   FROM clinic.model_results AS mr
                   JOIN clinic.studies AS s ON s.study_id = mr.study_id
                   JOIN clinic.authorized_cases AS ac ON ac.case_id = s.case_id
                   JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                   WHERE ac.user_id = %s AND ac.case_id = %s AND s.study_id = %s
                     AND p.is_synthetic AND mr.is_current
                     AND mr.record_status = 'active'
                   ORDER BY mr.model_result_id""",
                scope,
            )
            current_rows = cursor.fetchall()
            if len(current_rows) > 1:
                raise ClinicalReadError("RECORD_CONFLICT")
            current_model_result = dict(current_rows[0]) if current_rows else None
            cursor.execute(
                """SELECT mr.model_result_id, mr.model_version, mr.record_version, mr.is_current
                   FROM clinic.model_results AS mr
                   JOIN clinic.studies AS s ON s.study_id = mr.study_id
                   JOIN clinic.authorized_cases AS ac ON ac.case_id = s.case_id
                   JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                   WHERE ac.user_id = %s AND ac.case_id = %s AND s.study_id = %s
                     AND p.is_synthetic AND mr.inference_status = 'completed'
                     AND mr.record_status IN ('active', 'superseded')
                   ORDER BY mr.record_version, mr.model_result_id""",
                scope,
            )
            references = [dict(item) for item in cursor.fetchall()]
        return {
            **self._study_data(row, case),
            "current_model_result": current_model_result,
            "model_results": references,
        }, []

    def _get_model_result(self, args: Any, case: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        study = self._owned_study(case["case_id"], args.study_id)
        with self._connect() as connection, connection.cursor() as cursor:
            if args.model_result_id is None:
                cursor.execute(
                    """SELECT mr.* FROM clinic.model_results AS mr
                       JOIN clinic.studies AS s ON s.study_id = mr.study_id
                       JOIN clinic.authorized_cases AS ac ON ac.case_id = s.case_id
                       JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                       WHERE ac.user_id = %s AND ac.case_id = %s AND p.is_synthetic
                         AND s.study_id = %s AND mr.inference_status = 'completed'
                         AND mr.record_status = 'active' AND mr.is_current
                       ORDER BY mr.model_result_id""",
                    (self._user_id, case["case_id"], study["study_id"]),
                )
                rows = cursor.fetchall()
                if not rows:
                    raise ClinicalReadError("RESULT_NOT_AVAILABLE")
                if len(rows) != 1:
                    raise ClinicalReadError("RECORD_CONFLICT")
                row = rows[0]
            else:
                cursor.execute(
                    """SELECT mr.* FROM clinic.model_results AS mr
                       JOIN clinic.studies AS s ON s.study_id = mr.study_id
                       JOIN clinic.authorized_cases AS ac ON ac.case_id = s.case_id
                       JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                       WHERE ac.user_id = %s AND ac.case_id = %s AND p.is_synthetic
                         AND s.study_id = %s AND mr.model_result_id = %s""",
                    (self._user_id, case["case_id"], study["study_id"], args.model_result_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ClinicalReadError("RECORD_NOT_FOUND")
                if row["inference_status"] != "completed" or row["record_status"] not in ("active", "superseded"):
                    raise ClinicalReadError("RESULT_NOT_AVAILABLE")
            cursor.execute(
                """SELECT label, assessment, confidence FROM clinic.model_findings
                   WHERE model_result_id = %s ORDER BY finding_index""",
                (row["model_result_id"],),
            )
            findings = [dict(item) for item in cursor.fetchall()]
        data = {
            "case_id": case["case_id"], "patient_id": case["patient_id"],
            "study_id": study["study_id"], "model_result_id": row["model_result_id"],
            "model_name": row["model_name"], "model_version": row["model_version"],
            "generated_at": row["generated_at"], "is_current": row["is_current"],
            "review_status": "unreviewed", "findings": findings, "summary": row["summary"],
            "score_description": row["score_description"], "limitations": row["limitations"],
            "source": self._source("model_prediction", row["model_result_id"], row),
        }
        warnings = [] if row["is_current"] else [self._historical(row["model_result_id"], row)]
        return data, warnings

    def _get_reviewed_report(self, args: Any, case: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        study = self._owned_study(case["case_id"], args.study_id)
        with self._connect() as connection, connection.cursor() as cursor:
            if args.report_id is None:
                cursor.execute(
                    """SELECT rr.*, c.display_name AS reviewed_by_display_name
                       FROM clinic.reviewed_reports AS rr
                       JOIN clinic.studies AS s ON s.study_id = rr.study_id
                       JOIN clinic.authorized_cases AS ac ON ac.case_id = s.case_id
                       JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                       JOIN clinic.clinicians AS c ON c.clinician_id = rr.reviewed_by_clinician_id
                       WHERE ac.user_id = %s AND ac.case_id = %s AND p.is_synthetic
                         AND s.study_id = %s AND rr.review_status = 'reviewed'
                         AND rr.record_status = 'active' AND rr.is_current""",
                    (self._user_id, case["case_id"], study["study_id"]),
                )
                rows = cursor.fetchall()
                if len(rows) > 1:
                    raise ClinicalReadError("RECORD_CONFLICT")
                if not rows:
                    cursor.execute(
                        """SELECT 1 FROM clinic.reviewed_reports
                           WHERE study_id = %s AND review_status = 'draft'
                             AND record_status IN ('active', 'superseded') LIMIT 1""",
                        (study["study_id"],),
                    )
                    raise ClinicalReadError("NOT_REVIEWED" if cursor.fetchone() else "RECORD_NOT_FOUND")
                row = rows[0]
            else:
                cursor.execute(
                    """SELECT rr.*, c.display_name AS reviewed_by_display_name
                       FROM clinic.reviewed_reports AS rr
                       JOIN clinic.studies AS s ON s.study_id = rr.study_id
                       JOIN clinic.authorized_cases AS ac ON ac.case_id = s.case_id
                       JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                       LEFT JOIN clinic.clinicians AS c ON c.clinician_id = rr.reviewed_by_clinician_id
                       WHERE ac.user_id = %s AND ac.case_id = %s AND p.is_synthetic
                         AND s.study_id = %s AND rr.report_id = %s""",
                    (self._user_id, case["case_id"], study["study_id"], args.report_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ClinicalReadError("RECORD_NOT_FOUND")
                if row["review_status"] != "reviewed":
                    raise ClinicalReadError("NOT_REVIEWED")
                if row["record_status"] not in ("active", "superseded"):
                    raise ClinicalReadError("RECORD_NOT_FOUND")
        data = {
            "case_id": case["case_id"], "patient_id": case["patient_id"],
            "study_id": study["study_id"], "report_id": row["report_id"],
            "review_status": "reviewed", "is_current": row["is_current"],
            "reviewed_by": {"clinician_id": row["reviewed_by_clinician_id"], "display_name": row["reviewed_by_display_name"]},
            "reviewed_at": row["reviewed_at"], "findings_text": row["findings_text"],
            "impression_text": row["impression_text"],
            "follow_up_recommendation": row["follow_up_recommendation"],
            "supersedes_report_id": row["supersedes_report_id"],
            "source": self._source("reviewed_report", row["report_id"], row),
        }
        warnings = [] if row["is_current"] else [self._historical(row["report_id"], row)]
        return data, warnings

    def _get_appointments(self, args: Any, case: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT CURRENT_TIMESTAMP AS as_of")
            as_of = cursor.fetchone()["as_of"]
            clauses = ["ac.user_id = %s", "ac.case_id = %s", "p.is_synthetic"]
            params = [self._user_id, case["case_id"]]
            if args.time_scope == "upcoming":
                clauses.append("a.starts_at >= %s"); params.append(as_of)
            elif args.time_scope == "past":
                clauses.append("a.starts_at < %s"); params.append(as_of)
            if args.appointment_type is not None:
                clauses.append("a.appointment_type = %s"); params.append(args.appointment_type)
            if args.statuses is not None:
                clauses.append("a.status = ANY(%s)"); params.append(args.statuses)
            cursor.execute(
                """SELECT a.*, c.display_name AS clinician_display_name
                   FROM clinic.appointments AS a
                   JOIN clinic.authorized_cases AS ac ON ac.case_id = a.case_id
                   JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                   LEFT JOIN clinic.clinicians AS c ON c.clinician_id = a.clinician_id
                   WHERE """ + " AND ".join(clauses) + " ORDER BY a.starts_at, a.appointment_id",
                tuple(params),
            )
            rows = cursor.fetchall()
        appointments = []
        for row in rows:
            appointments.append({
                "appointment_id": row["appointment_id"], "appointment_type": row["appointment_type"],
                "starts_at": row["starts_at"].astimezone(ZoneInfo(row["timezone"])),
                "ends_at": None if row["ends_at"] is None else row["ends_at"].astimezone(ZoneInfo(row["timezone"])),
                "timezone": row["timezone"],
                "status": row["status"],
                "clinician": None if row["clinician_id"] is None else {"clinician_id": row["clinician_id"], "display_name": row["clinician_display_name"]},
                "location": row["location"], "notes": row["notes"],
                "replaces_appointment_id": row["replaces_appointment_id"],
                "source": self._source("appointment", row["appointment_id"], row),
            })
        return {"case_id": case["case_id"], "patient_id": case["patient_id"], "as_of": as_of, "appointments": appointments}, []

    def _owned_study(self, case_id: str, study_id: str) -> dict[str, Any]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT s.* FROM clinic.studies AS s
                   JOIN clinic.authorized_cases AS ac ON ac.case_id = s.case_id
                   JOIN clinic.patients AS p ON p.patient_id = ac.patient_id
                   WHERE ac.user_id = %s AND ac.case_id = %s AND s.study_id = %s
                     AND p.is_synthetic""",
                (self._user_id, case_id, study_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise ClinicalReadError("RECORD_NOT_FOUND")
        return dict(row)

    @staticmethod
    def _study_candidate(row: dict[str, Any]) -> dict[str, Any]:
        date = row["performed_at"].date().isoformat() if row["performed_at"] else "date unknown"
        return {"record_type": "study", "record_id": row["study_id"], "label": f"{date} {row['body_part']} {row['laterality']}"}

    def _study_data(self, row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
        return {"case_id": case["case_id"], "patient_id": case["patient_id"], "study_id": row["study_id"],
                "performed_at": row["performed_at"], "modality": row["modality"], "body_part": row["body_part"],
                "laterality": row["laterality"], "views": row["views"], "acquisition_status": row["acquisition_status"],
                "image_ref": row["image_ref"], "source": self._source("study", row["study_id"], row)}

    @staticmethod
    def _source(source_type: str, record_id: str, row: dict[str, Any]) -> dict[str, Any]:
        version = str(row["record_version"])
        updated_at = row["updated_at"]
        return {"evidence_id": f"{source_type}:{record_id}:v{version}:{updated_at.isoformat()}",
                "source_type": source_type, "record_id": record_id, "version": version, "updated_at": updated_at}

    def _historical(self, record_id: str, row: dict[str, Any]) -> dict[str, Any]:
        return {"code": "HISTORICAL_RECORD", "message": "Explicitly requested historical record.",
                "evidence_ids": [self._source("model_prediction" if "model_result_id" in row else "reviewed_report", record_id, row)["evidence_id"]]}
