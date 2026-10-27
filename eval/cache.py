"""
Verdicts kept across evaluations (paths.VERDICTS, SQLite): a check's result keyed by the check's kind, its identity
(tool versions, flags, the environment's packages) and the exact program text. Callers never store timeouts or crashes
of a checker; a result within a time limit is kept as first observed, at the load of that check.
"""

import functools
import hashlib
import importlib.metadata
import json
import sqlite3
import sys
import threading
from typing import Any

from paths import VERDICTS


@functools.cache
def environment() -> str:
    """The interpreter's version and every installed distribution's name and version."""
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()})
    return json.dumps([sys.version, packages])


class VerdictCache:
    """Results of one kind of check under one identity; get and put are thread-safe, and processes share the file."""
    lock = threading.Lock()
    connection: sqlite3.Connection | None = None

    def __init__(self, kind: str, identity: str):
        self.prefix = json.dumps([kind, identity, environment()]).encode() + b"\0"

    @classmethod
    def database(cls) -> sqlite3.Connection:
        """This process's connection (under lock): WAL, so concurrent evaluations read and write the same file."""
        if cls.connection is None:
            cls.connection = sqlite3.connect(VERDICTS, check_same_thread=False, isolation_level=None)
            cls.connection.execute("PRAGMA journal_mode=WAL")
            cls.connection.execute("PRAGMA synchronous=NORMAL")
            cls.connection.execute("CREATE TABLE IF NOT EXISTS verdicts (key BLOB PRIMARY KEY, value TEXT NOT NULL)")
        return cls.connection

    def key(self, payload: str) -> bytes:
        return hashlib.sha256(self.prefix + payload.encode("utf-8", "surrogatepass")).digest()

    def get(self, payloads: list[str]) -> list[Any]:
        """Each payload's stored result, None if absent."""
        keys = [self.key(p) for p in payloads]
        found = {}
        with self.lock:
            database = self.database()
            chunk = database.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)  # host parameters per statement
            for k in range(0, len(keys), chunk):
                part = keys[k:k + chunk]
                query = f"SELECT key, value FROM verdicts WHERE key IN ({','.join('?' * len(part))})"
                found.update(database.execute(query, part).fetchall())
        return [None if k not in found else json.loads(found[k]) for k in keys]

    def put(self, items: list[tuple[str, Any]]) -> None:
        """Store (payload, result) pairs; results must not be None."""
        rows = [(self.key(p), json.dumps(v)) for p, v in items]
        with self.lock:
            database = self.database()
            database.execute("BEGIN")
            database.executemany("INSERT OR REPLACE INTO verdicts VALUES (?, ?)", rows)
            database.execute("COMMIT")
