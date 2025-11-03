import sqlite3, json, os, time, argparse, logging
import pandas as pd, numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from pydantic import BaseModel, create_model
from typing import List, Literal, Tuple
import asyncio
from openai import AsyncOpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@dataclass
class Answer:
    reasoning: str
    answer: float

ornithologist_system_prompt = """We are determining whether paragraphs from the Reserve Bank of Australia's monetary policy publications have a hawkish or dovish sentiment.

Topic: {topic} ({description})"""

ornithologist_user_prompt = """Paragraph:
{paragraph}

Previous reasoning:
{previous_reasoning}

{qn_string}"""

question_prompt = """{question}
Possible answers: {possible_answers}"""

# database functions
# database functions
# Note these three are the same as in ornithologist-v2.py: should package up.
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

# Guidance functions
def as_at(dataframe, variable_col, date_ymd, knowledge_date_col="knowledge_date"):
    # assumption: DF is ordered.
    # assumption: date is ISO y-m-d
    return dataframe.query(f"{knowledge_date_col} <= '{date_ymd}'").tail(1)[[variable_col]].squeeze()

def cpi_guidance(cpi_ye_val):
    if cpi_ye_val < 2:
        return {"inflation_state": "below_target"}
    if cpi_ye_val > 3:
        return {"inflation_state": "above_target"} 
    return {"inflation_state": "on_target"} 

# Tree and API functions
def next_node(tree: dict, answers: list[str]) -> Tuple[list[dict], dict | None]:
    node = tree
    path = []
    for ans in answers:
        path.append({"question": node.get("question", ""), "answer": ans})
        node = node.get("children", {}).get(ans, None)
        if node is None:
            return None
    return path, node

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def ask_gpt_5_mini(system_prompt, user_prompt, pydantic_format, openai_client, cheap=False):
    response = await openai_client.with_options(timeout=900.0).responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        text_format=pydantic_format,
        service_tier="flex" if cheap else "auto",
    )
    return response.output_parsed

async def generate_answer(paragraph: str, node: dict, path: List[dict], external_guidance: dict, metadata: dict, openai_client, cheap=False, logger=None):
    # assumes not a terminal
    answers = list(node["children"].keys()) # here's where we'd also have to put in a NA answer
    QuestionResponseFormat = create_model('QuestionResponseFormat', reasoning=str, answer=Literal[tuple(answers)])

    system_prompt = ornithologist_system_prompt.format(topic=metadata["topic"], description=metadata["description"])

    node_question_fmt = node["question"].format_map(external_guidance)
    history = "\n".join(["* " + ans["question"] + " (Answer: " + ans["answer"] + ")" for ans in path])
    qn = question_prompt.format(question=node_question_fmt, possible_answers="; ".join(answers))
    user_prompt = ornithologist_user_prompt.format(paragraph=paragraph, qn_string=qn, previous_reasoning=history)
    if logger is not None:
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"User prompt: {user_prompt}")
    #return((system_prompt, user_prompt, node.get("type", "none")))
    
    if len(answers) == 0:
        node_type = "terminal"
    else:
        node_type = node.get("type", "LLM")
        
    if node_type == "SYMBOLIC":
        return QuestionResponseFormat(reasoning="symbolic", answer=node_question_fmt) # could pass in reasoning when we construct the guidance dict
    if node_type == "LLM":
        #ans = await ask_gpt_5_mini(node_question_fmt, paragraph, QuestionResponseFormat, openai_client, cheap=cheap)
        ans = await ask_gpt_5_mini(system_prompt, user_prompt, QuestionResponseFormat, openai_client, cheap=cheap)
        return ans

async def tree_done_answer():
    return None

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
    
async def ornithologist(doc_id, conn, client, cpi, trees, cheap=False, max_concurrent_llm: int | None = None, logger = None):
    """Run taxonomy-guided reasoning for a document.

    Concurrency model: Each (chunk_id, tree_tag) pair runs independently to completion, issuing sequential
    LLM calls as needed. A semaphore limits simultaneous LLM requests if max_concurrent_llm is set.
    """
    doc = get_document(conn, doc_id)
    doc_tags = get_document_tags(conn, doc_id)
    doc_date = doc.date.min()
    doc_guidance = cpi_guidance(as_at(cpi, "cpi_headline_ye", doc_date))
    doc_chunks = {row.chunk_id: row.chunk_text for idx, row in doc.iterrows()}

    # Prepare workload (chunk_id, text, tree_tag)
    workload = []
    for idx, row in doc_tags.iterrows():
        tag = row.tag
        if tag == "CORE-TRADEABLENONTRADEABLE":
            tag = "CORE-TRADABLENONTRADABLE"
        if tag in trees:
            workload.append((row.chunk_id, doc_chunks[row.chunk_id], tag))

    semaphore = asyncio.Semaphore(max_concurrent_llm) if (max_concurrent_llm and max_concurrent_llm > 0) else None

    async def run_tree(chunk_id: int, text: str, tree_tag: str):
        tree_path: list[Answer] = []
        while True:
            path, node = next_node(trees[tree_tag]["tree"], [x.answer for x in tree_path])
            if node is None or len(node.get("children", {})) == 0:  # terminal
                break
            if node.get("type", "LLM") == "SYMBOLIC":
                # symbolic node: deterministic
                ans_obj = Answer(reasoning="symbolic", answer=node["question"].format_map(doc_guidance))
                tree_path.append(ans_obj)
                continue
            # LLM node: possibly limited by semaphore
            async def ask():
                return await generate_answer(text, node, path, doc_guidance, trees[tree_tag], client, cheap=cheap, logger=logger)
            if semaphore:
                async with semaphore:
                    update = await ask()
            else:
                update = await ask()
            update_answer = Answer(reasoning=update.reasoning, answer=update.answer)
            tree_path.append(update_answer)
        # Acquire final assessment node
        final_path, assessment_node = next_node(trees[tree_tag]["tree"], [x.answer for x in tree_path])
        if assessment_node is None:
            assessment = "skip"
        else:
            assessment = assessment_node.get("assessment", "skip")
        return chunk_id, text, tree_tag, tree_path, final_path, assessment

    tasks = [asyncio.create_task(run_tree(chunk_id, text, tree_tag)) for chunk_id, text, tree_tag in workload]
    ornithologist_results: dict[int, dict] = {}
    for task in asyncio.as_completed(tasks):
        chunk_id, text, tree_tag, tree_path, final_path, assessment = await task
        if chunk_id not in ornithologist_results:
            ornithologist_results[chunk_id] = {"text": text, "taxonomy-guided-reasoning": {}}
        save_path = [{"question": p["question"], "reasoning": tp.reasoning, "answer": tp.answer} for p, tp in zip(final_path, tree_path)]
        ornithologist_results[chunk_id]["taxonomy-guided-reasoning"][tree_tag] = {"path": save_path, "assessment": assessment}

    # Scoring
    scores = {k: [get_category(tgr["assessment"]) for tgr in v["taxonomy-guided-reasoning"].values()] for k, v in ornithologist_results.items()}
    scores_para = {k: [x for x in v if x != -1] for k, v in scores.items()}
    scores_para_avg = {k: np.mean(v) for k, v in scores_para.items() if len(v) > 0}
    doc_score = np.mean([x for x in scores_para_avg.values()]) if scores_para_avg else float('nan')
    return {"results": ornithologist_results, "scores": scores, "scores_para_avg": scores_para_avg, "doc_score": doc_score }

def configure_logging(level: str):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

def parse_doc_ids_arg(s: str) -> list[int] | None:
    # Accept comma separated list or ranges like 1-5
    if s is None:
        return None
    ids = set()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            ids.update(range(a, b + 1))
        else:
            ids.add(int(p))
    return sorted(ids)

async def main(args):
    logger = logging.getLogger("ornithologist.reasoning")
    logger.info("Hello.")

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    # Optional reasoning cache table creation
    if args.sqlite:
        table = "reasoning_cache"
        cursor.execute("""
        create table if not exists reasoning_cache (
            doc_id integer primary key,
            reasoning_json text not null,
            updated_at text default (datetime('now'))
        )""")
        conn.commit()
        logger.info("SQLite reasoning cache ready (table=reasoning_cache).")
    llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("Database and API connected.")

    cpi = pd.read_csv(args.cpi_path)
    cpi.knowledge_date = pd.to_datetime(cpi.knowledge_date, dayfirst=True)
    logger.info("CPI guidance loaded.")

    with open(args.taxonomy_metadata, "r") as f:
        taxonomy_metadata = json.load(f)
    logger.info(f"Taxonomy metadata loaded. {len(taxonomy_metadata)} entries found.")

    trees = {}
    with os.scandir(args.tree_dir) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                tree_tag, _ = os.path.splitext(entry.name)
                with open(entry.path, "r") as f:
                    tree_json = json.load(f)
                trees[tree_tag] = {"tree": tree_json, "topic": taxonomy_metadata[tree_tag]["topic"], "description": taxonomy_metadata[tree_tag]["description"]}
    logger.info(f"Decision trees loaded. {len(trees)} entries found.")

    cursor.execute("select distinct doc_id from documents")
    all_doc_ids = [x[0] for x in cursor.fetchall()]
    logger.info(f"Document IDs loaded. {len(all_doc_ids)} entries found ({min(all_doc_ids)} - {max(all_doc_ids)})")

    # determine which doc ids to process
    doc_ids = None
    if args.doc_ids:
        doc_ids = parse_doc_ids_arg(args.doc_ids)
    if args.doc_range:
        doc_ids = parse_doc_ids_arg(args.doc_range)
    if doc_ids is None:
        doc_ids = all_doc_ids

    # make output dir
    os.makedirs(args.outdir or "ornithologist-output", exist_ok=True)

    # Run job
    start_time = time.time()
    # Decide which doc_ids require computation based on mode
    to_compute = []
    if args.sqlite:
        # In sqlite mode: recompute only if --recompute OR not cached yet
        cached_ids = set(row[0] for row in cursor.execute("select doc_id from reasoning_cache"))
        for doc_id in doc_ids:
            if args.recompute or doc_id not in cached_ids:
                to_compute.append(doc_id)
    else:
        # JSON filesystem mode: recompute if --recompute OR file missing
        for doc_id in doc_ids:
            outpath = os.path.join(args.outdir or "ornithologist-output", f"{doc_id}.json")
            if args.recompute or not os.path.isfile(outpath):
                to_compute.append(doc_id)

    logger.info(f"Docs to compute: {len(to_compute)} / {len(doc_ids)}")

    for doc_id in tqdm(to_compute, desc="Ornithologist"):
        outpath = os.path.join(args.outdir or "ornithologist-output", f"{doc_id}.json")
        res = await ornithologist(doc_id, conn, llm_client, cpi, trees, cheap=args.cheap, max_concurrent_llm=args.max_concurrent_llm, logger=logger)
        with open(outpath, "w") as f:
            json.dump(res, f)
        if args.sqlite:
            cursor.execute("insert into reasoning_cache (doc_id, reasoning_json, updated_at) values (?, ?, datetime('now')) on conflict(doc_id) do update set reasoning_json=excluded.reasoning_json, updated_at=datetime('now')", (doc_id, json.dumps(res)))
            conn.commit()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Completed in {elapsed_time:.2f} seconds")
    logger.info("Goodbye.")

def build_arg_parser():
    p = argparse.ArgumentParser(description="Run the ornithologist processing over documents")
    p.add_argument("--db-path", type=str, default="documents.db", help="Path to the SQLite database file.")
    p.add_argument("--cpi-path", type=str, default="data/tm_cpi_knowledge.csv", help="Path to the CPI knowledge CSV file.")
    p.add_argument("--taxonomy-metadata", type=str, default=None, help="Path to the taxonomy metadata JSON file.")
    p.add_argument("--tree-dir", type=str, default=None, help="Path to the directory containing tree JSON files.")
    p.add_argument("--doc-ids", type=str, default=None, help="Comma separated doc ids or ranges, e.g. 1,2,5-10")
    p.add_argument("--doc-range", type=str, default=None, help="Alias for --doc-ids using a single range like 10-20")
    p.add_argument("--outdir", type=str, default=None, help="Output directory for json files")
    p.add_argument("--recompute", help="Recompute existing outputs (JSON mode) or cache entries (SQLite mode)", action="store_true")
    p.add_argument("--cheap", help="Use cheap mode (OpenAI flex processing)", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    p.add_argument("--sqlite", help="Enable SQLite cache persistence and reuse (reasoning_cache table)", action="store_true")
    p.add_argument("--max-concurrent-llm", type=int, default=None, help="Maximum number of concurrent LLM requests (default: unlimited)")
    return p

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)
    asyncio.run(main(args))