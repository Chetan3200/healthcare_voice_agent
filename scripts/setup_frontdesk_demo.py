"""Prepare or inspect ONLY the isolated synthetic PostgreSQL front-desk demo."""
from __future__ import annotations

import argparse
import json

from healthcare_voice_agent.demo.database import (
    DemoDatabaseError, _alembic_head, build_demo_engine, link_existing_demo_booking,
    prepare_demo_database, verify_demo_database,
)

EXPECTED_SCHEMA_REVISION = _alembic_head()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Explicitly create/start the separate demo container, migrate, seed and publish slots")
    mode.add_argument("--link-existing", action="store_true", help="Data-only: link canonical AP-1042-01 provenance in the already prepared isolated demo")
    mode.add_argument("--check", action="store_true", help="Read-only demo readiness check (default)")
    args = parser.parse_args(argv)
    engine = None
    try:
        if args.prepare:
            result = prepare_demo_database()
        else:
            engine = build_demo_engine()
            result = link_existing_demo_booking(engine) if args.link_existing else verify_demo_database(engine)
        if result.get("revision") != EXPECTED_SCHEMA_REVISION:
            raise DemoDatabaseError("Demo schema revision does not match the packaged front-desk backend.")
        print(json.dumps(result, indent=2))
        print("Isolated synthetic demo only. The original clinic database was not selected.")
        return 0
    except DemoDatabaseError as exc:
        parser.exit(1, str(exc) + "\n")
    except Exception:
        parser.exit(1, "Demo readiness check failed. Run --prepare or check the isolated container; credentials were not printed.\n")
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
