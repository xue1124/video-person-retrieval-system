"""Create a new local account; prompt for a password without placing it in shell history."""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--role", choices=("admin", "user"), default="user")
    args = parser.parse_args()
    username = args.username.strip()
    if not username or len(username) > 64:
        parser.error("username must contain 1 to 64 characters")
    password = getpass.getpass("Password (at least 12 characters): ")
    if len(password) < 12 or len(password.encode("utf-8")) > 72:
        parser.error("password must contain at least 12 characters and at most 72 UTF-8 bytes")
    if getpass.getpass("Repeat password: ") != password:
        parser.error("passwords do not match")

    from db import get_engine
    from auth_deps import hash_password
    from sqlalchemy import text

    engine = get_engine()
    with engine.begin() as conn:
        if conn.execute(text("SELECT id FROM users WHERE username=:u"), {"u": username}).first():
            parser.error("username already exists; no account was changed")
        conn.execute(
            text("INSERT INTO users (username, password_hash, role, is_active) VALUES (:u, :p, :r, 1)"),
            {"u": username, "p": hash_password(password), "r": args.role},
        )
    engine.dispose()
    print("Account created. No password was printed or saved in plaintext.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
