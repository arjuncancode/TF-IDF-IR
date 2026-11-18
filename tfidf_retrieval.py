#!/usr/bin/env python3
"""
TF-IDF information retrieval mini-project.
Run from the project root: python src/tfidf_retrieval.py
Uses only the Python standard library.
"""
from pathlib import Path
from collections import Counter
import csv
import math
import re

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs"
STOPWORDS = set("""a an and are as at be but by can common from for help in into is it low manage of on or provides reduce reduces requires saves stations the to use well when with works""".split())

def tokenize(text):
    """Lowercase, remove punctuation/stopwords, and normalize simple plurals."""
    words = re.findall(r"[a-z]+", text.lower())
    result = []
    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        result.append(word)
    return result

def cosine_similarity(a, b):
    dot = sum(a.get(term, 0.0) * b.get(term, 0.0) for term in set(a) | set(b))
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def main():
    files = sorted(DATA_DIR.glob("*.txt"))
    if len(files) < 10:
        raise SystemExit("The collection must contain at least 10 text documents.")
    documents = {f.stem: f.read_text(encoding="utf-8") for f in files}
    tokens = {doc: tokenize(text) for doc, text in documents.items()}
    tf = {doc: Counter(items) for doc, items in tokens.items()}
    terms = sorted({term for counts in tf.values() for term in counts})
    n_docs = len(documents)
    df = {term: sum(term in counts for counts in tf.values()) for term in terms}
    idf = {term: math.log10(n_docs / df[term]) for term in terms}
    vectors = {
        doc: {term: count * idf[term] for term, count in counts.items()}
        for doc, counts in tf.items()
    }

    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "term_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "document_frequency", "idf_log10", *documents.keys()])
        for term in terms:
            writer.writerow([term, df[term], f"{idf[term]:.6f}",
                             *[tf[doc][term] for doc in documents]])

    with (OUT_DIR / "document_tfidf.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", *terms])
        for doc in documents:
            writer.writerow([doc, *[f"{vectors[doc].get(term, 0.0):.6f}" for term in terms]])

    queries = {
        "Q1 - solar wind energy": "solar wind energy",
        "Q2 - battery storage electricity": "battery storage electricity",
        "Q3 - nuclear carbon emissions": "nuclear carbon emissions",
        "Q4 - absent term": "quantum",
    }
    lines = [f"Documents: {n_docs}; vocabulary size: {len(terms)}",
             "IDF formula: log10(N / DF); TF-IDF: TF * IDF",
             "Ranking: cosine similarity between query and document TF-IDF vectors.", ""]
    for label, query in queries.items():
        q_counts = Counter(tokenize(query))
        q_vector = {term: count * idf[term] for term, count in q_counts.items() if term in idf}
        lines.append(f"{label} | query='{query}'")
        if not q_vector:
            lines.append("  No query terms occur in the collection; all scores are 0.000000.")
            lines.append("  " + ", ".join(f"{doc}: 0.000000" for doc in documents))
            lines.append("")
            continue
        scores = [(doc, cosine_similarity(q_vector, vectors[doc])) for doc in documents]
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        for rank, (doc, score) in enumerate(scores, 1):
            lines.append(f"  {rank:2d}. {doc}: {score:.6f}")
        lines.append("")
    output = "\n".join(lines)
    (OUT_DIR / "rankings.txt").write_text(output, encoding="utf-8")
    print(output)

if __name__ == "__main__":
    main()
