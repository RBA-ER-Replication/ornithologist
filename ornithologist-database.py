import pandas as pd, numpy as np
import sqlite3, json, os
import argparse
from tqdm import tqdm
from contextlib import contextmanager
import logging

# database functions
def get_document(conn, doc_id):
    cursor = conn.cursor()
    cursor.execute("""
select d.doc_id, d.shortname, d.source, d.date, d.metadata, dc.chunk_id, dc.chunk_order, c.chunk_text
from documents d
left join docs_chunks dc on d.doc_id = dc.doc_id
left join chunks c on dc.chunk_id = c.chunk_id
where d.doc_id = ?
order by dc.chunk_order asc""", (doc_id,))
    return pd.DataFrame(cursor.fetchall(), columns=["doc_id", "shortname", "source", "date", "metadata", "chunk_id", "chunk_order", "chunk_text"])

def get_document_tags(conn, doc_id):
    cursor = conn.cursor()
    cursor.execute("""
select d.doc_id, dc.chunk_id, dc.chunk_order, t.tag, t.computed_at, t.dist_from_max_rrf_score, t.final_relevance_check
from documents d
left join docs_chunks dc on d.doc_id = dc.doc_id
left join chunks c on dc.chunk_id = c.chunk_id
left join tags t on t.chunk_id = c.chunk_id
where d.doc_id = ?
order by dc.chunk_id asc, t.dist_from_max_rrf_score desc""", (doc_id,))
    return pd.DataFrame(cursor.fetchall(), columns=["doc_id", "chunk_id", "chunk_order", "tag", "computed_at", "rrf_dist", "final_relevance_check"])

def get_chunk(conn, chunk_id):
    cursor = conn.cursor()
    cursor.execute("""
select c.chunk_id, c.chunk_text
from chunks c
where c.chunk_id = ?""", (chunk_id,))
    return pd.DataFrame(cursor.fetchall(), columns=["chunk_id", "chunk_text"])

def create_new_database(conn):
    cursor = conn.cursor()
    # Schema (idempotent)
    cursor.execute('''CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY,
        shortname TEXT,
        source TEXT,
        date TEXT,
        metadata TEXT,
        filename TEXT
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY,
        chunk_text TEXT UNIQUE
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS docs_chunks (
        doc_id INTEGER,
        chunk_id INTEGER,
        chunk_order INTEGER,
        PRIMARY KEY (doc_id, chunk_id, chunk_order),
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
        FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS tags (
        chunk_id INTEGER,
        tag TEXT,
        computed_at TEXT,
        rrf_score REAL,
        dist_from_max_rrf_score REAL,
        final_relevance_check TEXT,
        PRIMARY KEY (chunk_id, tag, computed_at),
        FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
    )''')
    # Helpful indices for lookup speed (IF NOT EXISTS supported in modern SQLite for CREATE INDEX)
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)''')
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_docs_chunks_doc ON docs_chunks(doc_id)''')
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_docs_chunks_chunk ON docs_chunks(chunk_id)''')
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_tags_chunk ON tags(chunk_id)''')
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)''')
    conn.commit()

def set_sqlite_pragmas(conn, fast: bool):
    """Apply PRAGMAs to accelerate bulk ingest. If fast is False we keep safer defaults.
    Fast mode trades durability guarantees during the ingest for speed. Use only for one-off builds."""
    cursor = conn.cursor()
    if fast:
        # Aggressive settings – acceptable for throwaway build phases
        cursor.execute("PRAGMA journal_mode=OFF")
        cursor.execute("PRAGMA synchronous=OFF")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA locking_mode=EXCLUSIVE")
        cursor.execute("PRAGMA cache_size = -100000")  # ~100MB cache (negative => KB units)
    else:
        # Reasonable runtime defaults
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
    conn.commit()

@contextmanager
def bulk_transaction(conn):
    """Context manager ensuring a single large transaction for speed."""
    try:
        conn.execute('BEGIN')
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def add_document(conn, shortname, source, date, metadata, chunks, filename=None, commit: bool = True):
    """Insert a single document and its chunks.
    Setting commit=False lets callers batch many documents into one transaction for speed."""
    cursor = conn.cursor()
    cleaned_chunks = chunks # we could clean here but assume it's done on the json document side

    cursor.execute(
        "INSERT INTO documents (shortname, source, date, metadata, filename) VALUES (?, ?, ?, ?, ?) RETURNING doc_id",
        (shortname, source, date, json.dumps(metadata), "" if filename is None else filename),
    )
    doc_id = cursor.fetchone()[0]

    # Bulk insert (ignore duplicates) then resolve IDs.
    cursor.executemany(
        "INSERT OR IGNORE INTO chunks (chunk_text) VALUES (?)",
        [(text,) for text in cleaned_chunks]
    )

    # Use parameterised SELECT per chunk (avoids building huge IN clause strings for very large docs).
    # Empirically for moderate list sizes both are fine; this is more memory safe and simpler.
    chunk_id_lookup_stmt = "SELECT chunk_id FROM chunks WHERE chunk_text = ?"
    doc_chunk_map = []
    order = 0
    for text in cleaned_chunks:
        cursor.execute(chunk_id_lookup_stmt, (text,))
        chunk_id = cursor.fetchone()[0]
        doc_chunk_map.append((doc_id, chunk_id, order))
        order += 1
    cursor.executemany(
        "INSERT INTO docs_chunks (doc_id, chunk_id, chunk_order) VALUES (?, ?, ?)",
        doc_chunk_map
    )

    if commit:
        conn.commit()
    return doc_id

# Ingest from JSON files
def ingest_directory(conn, directory, skip_existing: bool = False, fast_mode: bool = False, logger: logging.Logger | None = None):
    """Ingest all JSON files in a directory.
    fast_mode => uses a single transaction & (optionally) aggressive PRAGMAs (applied outside this function).
    """
    doc_ids = {}
    if logger is None:
        logger = logging.getLogger(__name__)

    if skip_existing:
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM documents")
        existing_files = {row[0] for row in cursor.fetchall()}
    else:
        existing_files = set()

    json_files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() == ".json"]
    if skip_existing:
        # log all of the skipped files for debug
        for f in existing_files:
            if f in json_files:
                logger.debug(f"Skipped existing file: {f}")

        prev_length = len(json_files)
        json_files = [f for f in json_files if os.path.splitext(f)[0] not in existing_files]
        logger.info(f"Filtered JSON files: {prev_length} -> {len(json_files)}")
    iterator = tqdm(json_files, desc="Ingesting JSON", unit="file") if json_files else []

    # Wrap whole ingest in one transaction for speed if fast_mode.
    if fast_mode:
        ctx = bulk_transaction(conn)
    else:
        # No-op context manager
        @contextmanager
        def _noop():
            yield
        ctx = _noop()

    with ctx:
        for filename in iterator:
            file_basename, _ = os.path.splitext(filename)
            path = os.path.join(directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                paras = [x.strip() for x in data.get("fulltext", "").split("\n") if x.strip() != ""]
                doc_id = add_document(
                    conn,
                    data.get("shortname"),
                    data.get("source"),
                    data.get("date"),
                    data.get("metadata"),
                    paras,
                    file_basename,
                    commit=not fast_mode  # defer commit when in fast mode
                )
                doc_ids[file_basename] = doc_id
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
        # If not in fast_mode, commits happen per document; if in fast_mode, bulk_transaction handles final commit.
    if not fast_mode:
        conn.commit()
    return doc_ids

def configure_logging(level: str):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

def main(args):
    logger = logging.getLogger("ornithologist.ingest")
    logger.info("Hello.")

    conn = sqlite3.connect(args.db_path)
    create_new_database(conn)
    if args.fast_ingest:
        logger.warning("Applying fast ingest PRAGMAs (reduced durability during build)...")
        set_sqlite_pragmas(conn, fast=True)
    else:
        set_sqlite_pragmas(conn, fast=False)

    ids = ingest_directory(conn, args.corpus_dir, args.skip_existing, fast_mode=args.fast_ingest, logger=logger)
    logger.info(f"Ingested {len(ids)} documents.")

    logger.info("Goodbye.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the database.")
    parser.add_argument("--db-path", type=str, default="documents.db", help="Path to the SQLite database file.")
    parser.add_argument("--corpus-dir", type=str, default="corpus-import", help="Path to the directory containing the corpus files.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip ingesting documents that already exist in the database.")
    parser.add_argument("--fast-ingest", action="store_true", help="Use aggressive SQLite PRAGMAs and a single large transaction for speed (unsafe if interrupted).")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args()

    configure_logging(args.log_level)
    main(args)