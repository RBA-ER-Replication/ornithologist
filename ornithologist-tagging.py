import pandas as pd, networkx as nx
from functools import reduce
import sqlite3, json
from datetime import datetime
import argparse
from tqdm import tqdm
import logging

# Language
import spacy
from sentence_splitter import SentenceSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# Tagging functions
def sent_jaccard(sents, mode="jaccard"):
    ret = []
    for i, sent_a in enumerate(sents):
        for j, sent_b in enumerate(sents):
            if i == j:
                continue
            similarity = len(sent_a & sent_b) / len(sent_a | sent_b)
            if mode == "condprob":
                similarity = len(sent_a & sent_b) / len(sent_a)
            ret.append((i, j, similarity))
    return ret

def rank_sentences(sentences, spacy_nlp, mode="condprob"):
    if len(sentences) == 1:
        return {0: 1.0}
    lemmatised_sents = [{token.lemma_ for token in doc if not token.is_punct} for doc in spacy_nlp.pipe(sentences)]
    lemmatised_sents = pd.DataFrame(sent_jaccard(lemmatised_sents, mode), columns=["source", "target", "weight"])
    g = nx.from_pandas_edgelist(lemmatised_sents, "source", "target", "weight")
    ranks = nx.pagerank(g, weight="weight")
    return ranks

def taxonomy_classify(documents, taxonomy, crossencoder):
    # line up document, taxonomy surface representation pairs with metadata
    to_predict = []
    for doc_no, document in enumerate(documents):
        to_predict.extend( [((doc_no, k), [document, v]) for k, v in taxonomy.items()] )
    doc_info, doc_pairs = zip(*to_predict) # unzip
    scores = crossencoder.predict(list(doc_pairs))

    # match up the scores with metadata, group into useful format and return
    ret = {}
    for (doc_no, k), score in zip(doc_info, scores):
        if doc_no not in ret:
            ret[doc_no] = {}
        ret[doc_no][k] = score
    return [ret[i] for i in range(len(documents))]
    #return [(doc_no, k, score) for (doc_no, k), score in zip(doc_info, scores)]

def taxonomy_classify_sentences(documents, taxonomy, crossencoder, splitter, nlp):
    # split documents into sentences
    documents_sentences = [splitter.split(d) for d in documents]
    # line everything up
    to_predict = []
    doc_sent_ranks = {}
    for doc_no, doc_sents in enumerate(documents_sentences):
        doc_sent_ranks[doc_no] = rank_sentences(doc_sents, nlp)
        for k, v in taxonomy.items():
            to_predict.extend([ ((doc_no, sent_no, k), [sent, v]) for sent_no, sent in enumerate(doc_sents) ])
    doc_info, sent_pairs = zip(*to_predict) # unzip
    scores = crossencoder.predict(sent_pairs)

    # match up scores with metadata
    ret = {}
    for (doc_no, sent_no, k), score in zip(doc_info, scores):
        if doc_no not in ret:
            ret[doc_no] = {}
        if score >= 0.5 and k not in ret[doc_no]:
            ret[doc_no][k] = [(score, sent_no, doc_sent_ranks[doc_no][sent_no])]
        elif score >= 0.5 and k in ret[doc_no]:
            ret[doc_no][k].append((score, sent_no, doc_sent_ranks[doc_no][sent_no]))
    summarised = {}
    for doc_no, res in ret.items():
        summ = {}
        for k, v in res.items():
            sentence_votes = 0.0
            for (score, sent_no, doc_rank_score) in v:
                if score >= 0.5:
                    sentence_votes += doc_rank_score
            total_score = sum([score for (score, sent_no, doc_rank_score) in v])
            weighted_score = sum([score * doc_rank_score for (score, sent_no, doc_rank_score) in v])
            summ[k] = {"total": total_score, "weighted": weighted_score, "normalised_total": total_score/len(documents_sentences[doc_no]), "normalised_weighted": weighted_score/len(documents_sentences[doc_no]), "num_sentences_tagged": len(v), "total_sentences": len(documents_sentences[doc_no]), "votes": sentence_votes}
        summarised[doc_no] = summ
        #summarised[doc_no] = {k: (len(v), sum(v), sum(v)/len(documents_sentences[doc_no])) for k, v in res.items()}
    return summarised

def rrf(query_rankings, k=60, threshold=0.75): # weight it?
    scores = {}
    for qry_rank in query_rankings:
        for rank, element in enumerate(qry_rank):
            delta = 1.0 / (k + rank + 1)
            scores[element] = scores[element] + delta if element in scores else delta
    ret = [(el, score) for el, score in scores.items()]
    if len(ret) == 0: # just in case...
        return []
    ret.sort(key=lambda x: -x[1])
    max_score = ret[0][1]
    return [(el, sc, sc/max_score) for el, sc in ret if sc/max_score >= threshold]

def summarise_taxons(top_k_list):
    counts = {}
    n_taxons = 0
    for taxon in reduce(lambda x, y: x + y, top_k_list):
        counts[taxon] = counts[taxon] + 1 if taxon in counts else 1
        n_taxons += 1
    return sorted([(k, v/n_taxons, v) for k, v in counts.items()], key=lambda x: -x[1])

def classical_tag(texts, nlp, phrase_bm25, phrase_corpus, manual_classifications_dict, phrase_top_n=10):
    classically_processed_texts = [[tkn.lemma_ for tkn in doc] for doc in nlp.pipe(texts, disable=["tok2vec", "parser"])]
    phrase_results = [phrase_bm25.get_top_n(doc, phrase_corpus, n=phrase_top_n) for doc in classically_processed_texts]
    classical_ranks = [summarise_taxons([manual_classifications_dict[l] for l in res]) for res in phrase_results]
    return classical_ranks

def tag_all_paragraphs(paragraphs, taxonomy, crossencoder, splitter, nlp, phrase_bm25, phrase_corpus, manual_classifications_dict):
    para_level_tags = taxonomy_classify(paragraphs, taxonomy, crossencoder) # annoyingly doesn't return a list but a dict with index keys
    sentence_level_tags = taxonomy_classify_sentences(paragraphs, taxonomy, crossencoder, splitter, nlp)
    bm25_tags = classical_tag(paragraphs, nlp, phrase_bm25, phrase_corpus, manual_classifications_dict)

    para_level_ranked = [[e for e, _ in sorted([(topic, score) for topic, score in t.items() if score >= 0.5], key=lambda x: -x[1])] for t in para_level_tags]
    sentence_level_ranked = []
    for i in range(len(paragraphs)):
        sentence_level_sorted = sorted([(k, v["votes"]) for k, v in sentence_level_tags[i].items()], key=lambda x: -x[1])
        sentence_level_ranked.append([e for e, _ in sentence_level_sorted])
    bm25_ranked = [[topic_name for topic_name, frac, n in sorted(t, key=lambda x: -x[1]) if n > 1] for t in bm25_tags]

    ranked_tags = []
    for p, s, b in zip(para_level_ranked, sentence_level_ranked, bm25_ranked):
        ranked_tags.append(rrf([p, s, b], threshold=0.45))
    return ranked_tags

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def add_tags(conn, chunk_ids, tags, computed_at):
    cursor = conn.cursor()
    tag_data = []
    for chunk_id, ts in zip(chunk_ids, tags):
        for tagname, rrf_score, rrf_dist in ts:
            tag_data.append((chunk_id, tagname, computed_at, rrf_score, rrf_dist))
            
    cursor.executemany("INSERT INTO tags (chunk_id, tag, computed_at, rrf_score, dist_from_max_rrf_score) VALUES (?, ?, ?, ?, ?)",
                   tag_data)
    conn.commit()

def configure_logging(level: str):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

def main(args):
    logger = logging.getLogger("ornithologist.tagging")
    logger.info("Hello.")

    splitter = SentenceSplitter(language='en')
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded language utilities.")

    if args.fp16:
        crossencoder = CrossEncoder(args.crossencoder, num_labels=1, model_kwargs={"torch_dtype": "float16"})
        logger.info("Loaded crossencoder at FP16.")
    else:
        crossencoder = CrossEncoder(args.crossencoder, num_labels=1)
        logger.info("Loaded crossencoder.")
    logger.info(f"Crossencoder is on device {crossencoder.device}")

    with open(args.taxonomy_file, "r") as f:
        taxonomy = json.load(f)
    taxon_label_to_key = {v: k for k, v in taxonomy.items()}
    logger.info(f"Taxonomy loaded with {len(taxon_label_to_key)} entries.")

    manual_classifications = []
    with open(args.manual_classifications_file, "r") as f:
        for line in f:
            line_clean = line.strip().split("\t")
            manual_classifications.append((line_clean[0], [x.strip() for x in line_clean[1].split(",")]))
    manual_phrases = set([x[0] for x in manual_classifications])
    manual_classifications_dict = {k: v for k, v in manual_classifications}
    logger.info(f"{len(manual_phrases)} manual phrases loaded.")

    phrase_corpus = [phr.lower() for phr in manual_phrases]
    tokenized_phrase_corpus = [[tkn.lemma_ for tkn in doc] for doc in nlp.pipe(phrase_corpus, disable=["tok2vec", "parser"])]
    phrase_bm25 = BM25Okapi(tokenized_phrase_corpus)
    logger.info("BM25 index created.")

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    
    # get chunks that *don't* have tags:
    cursor.execute("SELECT chunk_id, chunk_text FROM chunks WHERE chunk_id NOT IN (SELECT chunk_id FROM tags)")
    untagged_chunks = cursor.fetchall()
    logger.info(f"Found {len(untagged_chunks)} untagged chunks.")

    if not untagged_chunks:
        logger.info("Skipping tagging.")
    else:
        # process in batches using the configured chunk size with a tqdm progress bar
        total_batches = (len(untagged_chunks) + args.chunk_size - 1) // args.chunk_size
        logger.info(f"Processing {len(untagged_chunks)} untagged chunks in {total_batches} batches.")
        batches = chunks(untagged_chunks, args.chunk_size)
        for i, batch in enumerate(tqdm(batches, total=total_batches, desc="Batches")):
            chunk_ids = [c[0] for c in batch]
            paragraphs = [c[1] for c in batch]  # ensure we pass a list of paragraph texts
            chunk_tags = tag_all_paragraphs(paragraphs, taxonomy, crossencoder, splitter, nlp, phrase_bm25, phrase_corpus, manual_classifications_dict)
            add_tags(conn, chunk_ids, chunk_tags, args.run_name)

    logger.info("Goodbye.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the database.")
    parser.add_argument("--db-path", type=str, default="documents.db", help="Path to the SQLite database file.")
    parser.add_argument("--taxonomy-file", type=str, default="data/taxonomy.json", help="Path to the taxonomy JSON file.")
    parser.add_argument("--manual-classifications-file", type=str, default="data/autophrase-classifications-manual.txt", help="Path to the manual classifications file.")
    parser.add_argument("--crossencoder", type=str, default="./crossencoder", help="Path to the crossencoder directory.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of paragraphs to pass to the cross-encoder at once.")
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Name of the run.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args()

    configure_logging(args.log_level)
    main(args)