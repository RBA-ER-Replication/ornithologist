import json, os, sqlite3, argparse, numpy as np
import pandas as pd

def get_category(assessment):
    assessment = assessment.lower()
    if "leaning dovish" in assessment:
        return 2
    elif "dovish" in assessment:
        return 1
    elif "neutral" in assessment:
        return 3
    elif "leaning hawkish" in assessment:
        return 4
    elif "hawkish" in assessment:
        return 5
    elif "skip" in assessment:
        return -1
    else:
        raise ValueError(f"Cannot categorise {assessment}")

def compute_analytical_score(json_data, topics: set, exclude_neutrals: bool = False):
    ret = []
    for chunk_id, result in json_data.get("results").items():
        #text = result.get("text")
        tgr = result.get("taxonomy-guided-reasoning")
        para_results = [get_category(tgr.get(topic, {}).get("assessment", "skip")) for topic in topics]
        para_results = [x for x in para_results if x != -1]
        if exclude_neutrals:
            para_results = [x for x in para_results if x != 3]
        if len(para_results) > 0:
            avg_score = np.mean(para_results)
            ret.append({"chunk_id": chunk_id, "average_score": avg_score})
    return np.mean([x["average_score"] for x in ret]) if len(ret) > 0 else np.nan

def read_scores(json_dir, analytical: set | None = None, analytical_exclude_neutrals: bool = False):
    scores = {}
    with os.scandir(json_dir) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                doc_id_str, extension = os.path.splitext(entry.name)
                if extension == ".json":
                    doc_id = int(doc_id_str)
                    with open(entry.path, "r") as f:
                        data = json.load(f)
                    if analytical is None:
                        scores[doc_id] = data.get("doc_score")
                    else:
                        scores[doc_id] = compute_analytical_score(data, analytical, analytical_exclude_neutrals)
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
    parser.add_argument("--analytical", help="List of topic tags to include in analysis")
    parser.add_argument("--analytical-exclude-neutrals", action="store_true", help="Exclude neutral assessments from analysis (analytical only)")
    args = parser.parse_args()

    if args.analytical:
        topics = set([x.strip() for x in args.analytical.split(",")])
        print(f"Analytical topics: {topics}")
    else:
        topics = None

    scores = read_scores(args.json_dir, analytical=topics, analytical_exclude_neutrals=args.analytical_exclude_neutrals)
    merged = join_with_db(args.db, scores)
    merged.to_csv(args.output, index=True)