"""Offline fixture tests and mocked transaction-control tests. Never open a DB."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
from datetime import timezone
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import seed


class FakeCursor:
    def __init__(self, rowcount=1, row=None):
        self.rowcount = rowcount
        self.row = row
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, values=None):
        self.statements.append(query if isinstance(query, str) else query.as_string())

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor
        self.closed = False

    def cursor(self):
        return self.fake_cursor

    def close(self):
        self.closed = True


class SeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = seed.load_dataset(seed.DEFAULT_FIXTURE_DIR)
        seed.validate_dataset(cls.dataset)

    def fresh(self):
        return copy.deepcopy(self.dataset)

    def test_full_dataset_passes(self):
        seed.validate_dataset(self.fresh())
        self.assertEqual(sum(map(len, self.dataset['tables'].values())), 288)
        self.assertEqual(len(self.dataset['tables']['cases']), 31)
        self.assertEqual(len(self.dataset['tables']['studies']), 45)

    def test_doctor_names_are_natural_and_still_explicitly_synthetic(self):
        rows = self.dataset['tables']['clinicians']
        self.assertEqual({row['clinician_id']: row['display_name'] for row in rows}, {
            'SYN-CLIN-01': 'Dr Mira Desai',
            'SYN-CLIN-02': 'Dr Dev Kapoor',
            'SYN-CLIN-03': 'Dr Leena Rao',
        })
        self.assertTrue(all(row['is_synthetic'] for row in rows))
        patient_names = {row['display_name'] for row in self.dataset['tables']['patients']}
        self.assertTrue(all(row['display_name'].removeprefix('Dr ') not in patient_names for row in rows))

    def test_loading_is_repeatable(self):
        self.assertEqual(self.dataset, seed.load_dataset(seed.DEFAULT_FIXTURE_DIR))

    def test_default_mode_never_connects_or_reads_environment(self):
        with patch.object(seed, '_connect', side_effect=AssertionError('Unexpected DB access')):
            with patch.object(seed, '_load_db_environment', side_effect=AssertionError('Unexpected .env read')):
                with redirect_stdout(io.StringIO()) as output:
                    result = seed.main([])
        self.assertEqual(result, 0)
        self.assertIn('Dry run OK', output.getvalue())

    def test_changed_fixture_bytes_fail_checksum(self):
        original_read = Path.read_bytes
        def altered_read(path):
            data = original_read(path)
            return data + b' ' if path.name == 'clinical_records.json' else data
        with patch.object(Path, 'read_bytes', altered_read):
            with self.assertRaises(seed.SeedError):
                seed.load_dataset(seed.DEFAULT_FIXTURE_DIR)

    def test_unknown_columns_are_rejected(self):
        dataset = self.fresh()
        dataset['tables']['patients'][0]['unrecognized_field'] = 'not allowed'
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_orphan_study_is_rejected(self):
        dataset = self.fresh()
        dataset['tables']['studies'][0]['case_id'] = '9999'
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_numeric_id_is_rejected(self):
        dataset = self.fresh()
        dataset['tables']['cases'][0]['case_id'] = 1042
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_naive_timestamp_is_rejected(self):
        dataset = self.fresh()
        dataset['tables']['studies'][0]['performed_at'] = '2026-09-10T08:15:00'
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_bad_confidence_is_rejected(self):
        dataset = self.fresh()
        dataset['tables']['model_findings'][0]['confidence'] = 1.2
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_duplicate_current_result_is_rejected(self):
        dataset = self.fresh()
        historical = next(row for row in dataset['tables']['model_results'] if row['model_result_id'] == 'MR-1054-01-V1')
        historical['is_current'] = True
        historical['record_status'] = 'active'
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_reviewed_report_requires_reviewer(self):
        dataset = self.fresh()
        dataset['tables']['reviewed_reports'][0]['reviewed_by_clinician_id'] = None
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_cross_case_appointment_replacement_is_rejected(self):
        dataset = self.fresh()
        dataset['tables']['appointments'][0]['replaces_appointment_id'] = 'AP-1043-01'
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_unknown_timezone_is_rejected(self):
        dataset = self.fresh()
        dataset['tables']['appointments'][0]['timezone'] = 'Moon/Imaginary'
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_nonverbatim_document_chunk_is_rejected(self):
        dataset = self.fresh()
        dataset['tables']['document_chunks'][0]['content'] = 'An invented passage absent from the source.'
        with self.assertRaises(seed.SeedError):
            seed.validate_dataset(dataset)

    def test_no_embeddings_are_seeded(self):
        self.assertEqual(self.dataset['tables']['retrieval_indexes'], [])
        self.assertEqual(self.dataset['tables']['chunk_embeddings'], [])

    def test_invalid_candidate_does_not_change_normal_rows(self):
        before = self.fresh()
        candidate = seed._invalid_candidate(self.dataset, self.dataset['invalid_records'][0])
        self.assertEqual(candidate['case_id'], '9999')
        self.assertEqual(self.dataset, before)

    def test_all_record_and_document_count_checks_pass(self):
        self.assertEqual(len(seed._validate_scenarios(self.dataset)), 92)

    def test_history_rows_are_sorted_parent_first(self):
        rows = list(reversed(self.dataset['tables']['reviewed_reports']))
        ordered = seed._topological_rows(rows, 'report_id', 'supersedes_report_id')
        positions = {row['report_id']: index for index, row in enumerate(ordered)}
        self.assertLess(positions['RR-1060-01-V1'], positions['RR-1060-01-V2'])

    def test_database_timestamp_normalization_matches(self):
        row = self.dataset['tables']['appointments'][0]
        spec = seed.TABLE_SPECS['appointments']
        values = [seed._db_value('appointments', key, row[key]) for key in spec.columns]
        values[spec.columns.index('starts_at')] = values[spec.columns.index('starts_at')].astimezone(timezone.utc)
        self.assertTrue(seed._row_matches('appointments', row, values))

    def test_identical_existing_row_is_skipped(self):
        row = self.dataset['tables']['patients'][0]
        values = tuple(seed._db_value('patients', key, row[key]) for key in seed.TABLE_SPECS['patients'].columns)
        cursor = FakeCursor(rowcount=0, row=values)
        self.assertEqual(seed._insert_or_compare(cursor, 'patients', row), 'skipped')
        self.assertFalse(any('UPDATE ' in statement or 'DELETE ' in statement for statement in cursor.statements))

    def test_different_existing_row_is_not_overwritten(self):
        row = self.dataset['tables']['patients'][0]
        columns = seed.TABLE_SPECS['patients'].columns
        values = [seed._db_value('patients', key, row[key]) for key in columns]
        values[columns.index('display_name')] = 'A changed existing name'
        cursor = FakeCursor(rowcount=0, row=tuple(values))
        with self.assertRaises(seed.SeedError):
            seed._insert_or_compare(cursor, 'patients', row)
        self.assertFalse(any('UPDATE ' in statement or 'DELETE ' in statement for statement in cursor.statements))

    def test_apply_commits_once_after_all_rows_and_checks(self):
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        with patch.object(seed, '_connect', return_value=connection), patch.object(seed, '_assert_revision'), patch.object(seed, '_insert_or_compare', return_value='inserted'), patch.object(seed, '_run_db_checks', return_value=92):
            self.assertEqual(seed.apply_dataset(self.dataset), (288, 0, 92))
        self.assertEqual(cursor.statements.count('COMMIT'), 1)
        self.assertNotIn('ROLLBACK', cursor.statements)
        self.assertTrue(connection.closed)

    def test_apply_rolls_back_on_a_row_conflict(self):
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        with patch.object(seed, '_connect', return_value=connection), patch.object(seed, '_assert_revision'), patch.object(seed, '_insert_or_compare', side_effect=['inserted', seed.SeedError('Fixture conflict')]):
            with self.assertRaises(seed.SeedError):
                seed.apply_dataset(self.dataset)
        self.assertIn('ROLLBACK', cursor.statements)
        self.assertNotIn('COMMIT', cursor.statements)
        self.assertTrue(connection.closed)

    def test_unexpected_invalid_insert_success_is_still_rolled_back(self):
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        with patch.object(seed, '_connect', return_value=connection), patch.object(seed, '_assert_revision'), patch.object(seed, '_verify_fixture_rows', return_value=288), patch.object(seed, '_run_db_checks', return_value=92):
            with self.assertRaises(seed.SeedError):
                seed.check_invalid_database(self.dataset)
        self.assertTrue(any(statement.startswith('ROLLBACK TO SAVEPOINT') for statement in cursor.statements))
        self.assertEqual(cursor.statements[-1], 'ROLLBACK')
        self.assertNotIn('COMMIT', cursor.statements)
        self.assertTrue(connection.closed)


if __name__ == '__main__':
    unittest.main()
