"""Create local database settings without printing or embedding credentials.

Python 3.10+. Uses only the standard library. Existing .env files are preserved.
"""

from pathlib import Path
import os
import secrets


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    target = ROOT / ".env"
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print("Existing .env preserved. No settings or credentials were changed.")
        return

    settings = {
        "POSTGRES_DB": "fracture_clinic",
        "POSTGRES_USER": "clinic_owner",
        "POSTGRES_PASSWORD": secrets.token_hex(24),
        "DB_HOST": "127.0.0.1",
        "DB_PORT": "5432",
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("# Local development settings. Never commit this file.\n")
        for key, value in settings.items():
            handle.write(f"{key}={value}\n")

    print("Created .env with a generated local password. No credentials were printed.")


if __name__ == "__main__":
    main()
