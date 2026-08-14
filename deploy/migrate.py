#!/usr/bin/env python3
"""Apply the SQL migrations in ``migrations/`` — spec 10, 17.

Each file runs once inside a transaction and is recorded in
``schema_migrations``, so re-running is a no-op. Migrations are applied at
deploy time, never by a worker on startup.

Usage:
    python deploy/migrate.py --status
    python deploy/migrate.py            # apply everything pending
    DATABASE_URL=postgresql://... python deploy/migrate.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = REPO_ROOT / "migrations"
DEFAULT_URL = "postgresql://sastt:sastt_dev@127.0.0.1:5432/sastt"

BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   TEXT PRIMARY KEY,
    sha256     TEXT        NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="list state and exit")
    args = parser.parse_args()

    try:
        import psycopg
    except ImportError:  # pragma: no cover - deployment dependency
        print("psycopg is not installed: pip install 'psycopg[binary,pool]'", file=sys.stderr)
        return 2

    with psycopg.connect(database_url(), autocommit=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute(BOOTSTRAP)
            cursor.execute("SELECT filename, sha256 FROM schema_migrations")
            applied = dict(cursor.fetchall())
        connection.commit()

        pending: list[Path] = []
        for path in migration_files():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            recorded = applied.get(path.name)
            if recorded is None:
                pending.append(path)
                state = "pending"
            elif recorded != digest:
                # An applied migration must never be edited: the database and
                # the file would silently disagree.
                print(f"[FAIL] {path.name}: already applied with a different checksum")
                return 1
            else:
                state = "applied"
            if args.status:
                print(f"{state:8s} {path.name}")

        if args.status:
            return 0
        if not pending:
            print("nothing to apply")
            return 0

        for path in pending:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"[apply] {path.name}")
            with connection.cursor() as cursor:
                cursor.execute(path.read_text(encoding="utf-8"))
                cursor.execute(
                    "INSERT INTO schema_migrations (filename, sha256) VALUES (%s, %s)",
                    (path.name, digest),
                )
            connection.commit()
        print(f"applied {len(pending)} migration(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
