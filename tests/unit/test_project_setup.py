"""Offline checks for the reorganized project, package, and tool contracts."""

from pathlib import Path

from healthcare_voice_agent.tools.contracts import TOOLS, validate_example_bundle
from scripts import check_db, seed, setup_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_scripts_resolve_the_repository_root():
    assert seed.ROOT == check_db.ROOT == setup_env.ROOT == PROJECT_ROOT
    assert seed.DEFAULT_FIXTURE_DIR == PROJECT_ROOT / "fixtures"


def test_all_five_tool_contracts_are_available():
    assert set(TOOLS) == {
        "get_study", "get_model_result", "get_reviewed_report",
        "get_appointments", "search_clinic_instructions",
    }


def test_all_ten_companion_examples_validate():
    examples = PROJECT_ROOT / "docs/contracts/fracture-clinic-tool-examples.json"
    assert validate_example_bundle(str(examples)) == 10


def test_initial_migration_is_still_present():
    assert (PROJECT_ROOT / "migrations/versions/001_initial_schema.py").is_file()
