# TF-IDF in Action: Build, Compute, Rank and Explain

## Project overview
This project builds a small information-retrieval collection of **10 original, short documents** about renewable energy and electricity systems. It implements tokenization, stop-word removal, simple plural normalization, TF, DF, IDF, TF-IDF, and cosine-similarity ranking.

## Folder structure
- `dataset/` — the ten plain-text documents used in every experiment.
- `src/tfidf_retrieval.py` — complete, standard-library Python implementation.
- `outputs/term_statistics.csv` — vocabulary-wide TF, DF and IDF table.
- `outputs/document_tfidf.csv` — complete document-vector matrix.
- `outputs/rankings.txt` — ranked output for four queries.
- `report.docx` — written assessment report and manual worked example.

## Run it
Python 3.9 or later is recommended. No third-party packages are required.

```bash
python src/tfidf_retrieval.py
```

Run from the project root. The program reads `dataset/` and regenerates the files in `outputs/`.

## Method
1. Lowercase and split text into alphabetic tokens.
2. Remove the explicitly listed stop words.
3. Normalize simple plurals (`batteries` → `battery`, `panels` → `panel`, etc.).
4. Use raw term frequency: `TF(t,d) = count(t,d)`.
5. Use document frequency: number of documents containing the term.
6. Use smoothed? **No** — IDF is `log10(N / DF(t))`, where `N=10`.
7. Weight each document term as `TF × IDF`.
8. Build query vectors with the same weighting and rank using cosine similarity.

## Reproducibility note
The CSV files are included as generated outputs. Running the script recomputes them from the supplied dataset. The collection is a small teaching corpus authored for this exercise, not a representative or statistically large real-world benchmark.

## Suggested GitHub workflow
Create a new GitHub repository, upload the contents of this folder (not just the ZIP), and commit. Then submit the repository URL and `report.docx` through Google Classroom, following your instructor's submission instructions.

## Academic integrity
Use this as a worked project and make sure you understand the calculations. Check your course's AI-use rules, make any changes your instructor expects, and acknowledge assistance if required. The report should be reviewed and personalized before submission.
