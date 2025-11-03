import json, os, sqlite3, argparse
import pandas as pd

def read_scores(json_dir):
    scores = {}
    with os.scandir(json_dir) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                doc_id_str, extension = os.path.splitext(entry.name)
                if extension == ".json":
                    doc_id = int(doc_id_str)
                    with open(entry.path, "r") as f:
                        data = json.load(f)
                    scores[doc_id] = data.get("doc_score")
    return scores

# line up with document IDs from the database
def join_with_db(db, scores):
    conn = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT doc_id, date, source, shortname FROM documents", conn)
    conn.close()

    # Merge scores with database document IDs
    merged = df.set_index("doc_id").join(pd.Series(scores, name="doc_score"), how="outer")
    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate document scores")
    parser.add_argument("--json-dir", help="Directory containing JSON files")
    parser.add_argument("--db", help="Path to the SQLite database")
    parser.add_argument("--output", default="collated_scores.csv", help="Path to the output CSV file")
    args = parser.parse_args()

    scores = read_scores(args.json_dir)
    merged = join_with_db(args.db, scores)
    merged.to_csv(args.output, index=True)