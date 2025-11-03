# Ornithologist: Towards Trustworthy "Reasoning" about Central Bank Communications
Dominic Zaun Eu Jones

I develop Ornithologist, a weakly-supervised textual classification system and measure the hawkishness and dovishness of central bank text. Ornithologist uses ``taxonomy-guided reasoning'', guiding a large language model with human-authored decision trees. This increases the transparency and explainability of the system and makes it accessible to non-experts. It also reduces hallucination risk. Since it requires less supervision than traditional classification systems, it can more easily be applied to other problems or sources of text (e.g. news) without much modification. Ornithologist measurements of hawkishness and dovishness of RBA communication carry information about the future of the cash rate path and of market expectations.

See: [Dominic Zaun Eu Jones. 2025. Ornithologist: Towards Trustworthy "Reasoning" about Central Bank Communications. arXiv: https://arxiv.org/abs/2505.09083](https://arxiv.org/abs/2505.09083)

**Note:** this project contains decision trees that assess the hawkishness or dovishness of central bank communication. These **do not** constitute the views of the Reserve Bank of Australia. Any views expressed are those of the author, not of the RBA.

## Overview
Core scripts:
- `ornithologist-database.py` – Build / update the SQLite database from JSON corpus.
- `ornithologist-tagging.py` – Assign topic tags to previously untagged paragraph chunks.
- `ornithologist-reasoning.py` – Run taxonomy-guided reasoning trees using tags + CPI guidance.

## SQLite database schema
- `documents(doc_id, shortname, source, date, metadata, filename)`
- `chunks(chunk_id, chunk_text UNIQUE)` – Paragraph-level text, deduplicated.
- `docs_chunks(doc_id, chunk_id, chunk_order)` – Ordering + many-to-many link.
- `tags(chunk_id, tag, computed_at, rrf_score, dist_from_max_rrf_score, final_relevance_check)`
- Optional: `reasoning_cache(doc_id, reasoning_json, updated_at)` when `--sqlite` used in reasoning.

## Installation
Tested on Python 3.10.
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# alternatively pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
```
Add OpenAI key (shell session):
```bash
export OPENAI_API_KEY="sk-..."
```

## Usage
### 1. Ingest
```bash
python ornithologist-database.py --corpus-dir corpus-import --db-path documents.db --fast-ingest
```
Omit `--fast-ingest` for safer settings.

### 2. Tag
Ensure taxonomy + manual classifications exist in `data/`.
```bash
python ornithologist-tagging.py \
  --db-path documents.db \
  --taxonomy-file data/taxonomy.json \
  --manual-classifications-file data/autophrase-classifications-manual.txt \
  --crossencoder ./crossencoder \
  --chunk-size 100 \
  --run-name $(date +%Y-%m-%d) \
  --fp16
```
Use `--fp16` to load the CrossEncoder in half precision (lower VRAM, faster throughput on GPU). Omit it for full precision.
If rerunning after edits, delete rows from `tags` for affected chunks or drop table.

### 3. Reason
```bash
python ornithologist-reasoning.py \
  --db-path documents.db \
  --cpi-path data/tm_cpi_knowledge.csv \
  --taxonomy-metadata data/taxonomy-metadata.json \
  --tree-dir data/decision-trees \
  --outdir ornithologist-output \
  --doc-range 1-50 \
  --cheap \
  --sqlite
```
Use `--recompute` to force regeneration. Omit `--sqlite` to write JSON files only.

### 4. Reporting

The tooling in `tooling/output-renderer` will render output JSON in an interpretable fashion. If you'd like to get scores across all documents, use the `collate-scores.py` script:

```bash
python collate-scores.py --json-dir out --db ornithologist.db --output scores.csv
```

## Tooling

The `tooling` directory contains three tools -- a decision tree editor, a database viewer/editor, and a final report renderer. You shouldn't need anything beyond a browser to run and use these. They were developed with the support of an AI-powered coding assistant.

## Performance
- Ingest: `--fast-ingest` applies aggressive PRAGMAs for initial bulk loads.
- Tagging: Increase `--chunk-size` cautiously (GPU/VRAM tradeoff). Reduce if OOM.
- Tagging FP16: `--fp16` halves crossencoder precision. See [speeding up inference](https://sbert.net/docs/cross_encoder/usage/efficiency.html).
- Reasoning: Limit parallel LLM calls with `--max-concurrent-llm` to manage rate limits.

## License

Code: Licensed under the BSD 3-Clause License.
Data under the `data` folder: Licensed under Creative Commons Attribution 4.0 International (CC-BY-4.0)
RBA Board minutes and statements corpus under the `corpus-import` folder is via the RBA (see [copyright details](https://www.rba.gov.au/copyright/)).
