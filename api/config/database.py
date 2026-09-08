from typing import Annotated
from fastapi import Depends
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy import event
from pathlib import Path

# Get the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(exist_ok=True)

sqlite_file_name = DB_DIR / "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
# check_same_thread=False — required: sessions get handed to worker threads
#   (see api/utils/executors.py) rather than only ever touched on the thread
#   that created them.
# timeout=30 — how long a connection waits for SQLite's file lock before
#   raising "database is locked", instead of pysqlite's 5s default. With a
#   few dozen concurrent users occasionally writing (logins, activity logs),
#   a short wait-and-succeed is what we want, not a fast, spurious failure.
connect_args = {"check_same_thread": False, "timeout": 30}

engine = create_engine(sqlite_url, connect_args=connect_args)


# WAL (Write-Ahead Logging) mode lets readers proceed while a write is in
# progress, instead of every read queuing up behind every write — this is
# the single biggest lever for SQLite under concurrent access, and costs
# nothing to enable. synchronous=NORMAL is the standard, safe pairing with
# WAL (still fsyncs at checkpoints; only relaxes per-transaction fsync,
# which is what makes WAL fast) — appropriate here since durability against
# an OS-level crash isn't a strict requirement for this app's data.
@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
       
SessionDep = Annotated[Session, Depends(get_session)]
