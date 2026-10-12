# Lab 2: Grounding & Context Engineering (RAG to GraphRAG)

**Course:** Intelligent System Development (知能システム開発特論 2026)  
**Affiliation:** Iwate Prefectural University (IPU)  
**Organization:** `ipu-intelligent-systems-2026`

---

## Objectives
Build beyond naive vector lookups. You will complete four TODOs in `lab2_starter.py`:

| TODO | What you implement | Lecture slide |
|------|--------------------|---------------|
| 1 | BM25 Okapi scoring (exact keywords, part numbers, error codes) | Slide 7 |
| 2 | Reciprocal Rank Fusion (RRF) of dense + sparse rankings | Slide 8 |
| 3 | Cross-encoder style reranker (100 candidates → best 3) | Slide 9 |
| 4 | Knowledge triple extraction (Subject, Predicate, Object) with Pydantic, then a NetworkX graph | Slide 11 |

The toy dense retriever, `build_graph`, and `main()` are already provided. Do not edit them.

## Files
- `lab2_starter.py` — your code (edit the four TODO blocks only)
- `sample_corpus.json` — 10 short documents, including error codes (`ERR-4039`), a part number (`XJ-4021`) and a paraphrase example (automobile maintenance)
- `tests/test_lab2.py` — the automated checks
- `requirements.txt`, `pytest.ini`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python lab2_starter.py
```
The program runs from the start, but the TODO functions return placeholder values (zero scores, empty triples). When you finish, you should see:
- Query `ERR-4039`: BM25 finds `doc_006`, while the dense list finds nothing (exact IDs are its weakness).
- Query `car repair tips`: the dense list finds `doc_005`, while BM25 finds nothing (paraphrase is BM25's weakness).
- RRF keeps the strengths of both, and the triples form a small graph.

## Verify
```bash
pytest
```
Work through the failing tests one TODO at a time. The RRF tests reproduce the table from Slide 8 (A = 0.0323, B = 0.0325, C = 0.0159).

## Submission
```bash
git add .
git commit -m "Complete Lab 2 hybrid retrieval and triple extraction"
git push origin main
```
Each push runs `pytest` through GitHub Actions. Your push is your official submission for Week 2.
