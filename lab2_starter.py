"""
Lab 2: Grounding & Context Engineering (Hybrid Search to Knowledge Triples)
Course: Intelligent System Development (知能システム開発特論)

Complete TODO 1-4 in this file:
  TODO 1  BM25 Okapi scoring             (Slide 7:  sparse search)
  TODO 2  Reciprocal Rank Fusion (RRF)   (Slide 8:  A/B/C table)
  TODO 3  Cross-encoder style reranker   (Slide 9:  100 -> 3 funnel)
  TODO 4  Knowledge triple extraction    (Slide 11: Paper X -> Aiko -> Prof. Sato)

Already provided (do not edit): the toy dense retriever, graph builder, and main().
Check your work with:  pytest
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field


# ==============================================================================
# Part 1: Schema Definitions for Graph Triples
# ==============================================================================

class EntityTriple(BaseModel):
    """Structured knowledge triple representing an extracted semantic relationship."""
    subject: str = Field(description="The primary entity or actor")
    predicate: str = Field(description="The relationship or directional action")
    object: str = Field(description="The target entity, property, or concept")


class ExtractionResult(BaseModel):
    """Container for document extraction."""
    document_id: str
    triples: List[EntityTriple] = Field(default_factory=list)


# ==============================================================================
# Part 2: BM25 Lexical Retrieval Engine
# ==============================================================================

class BM25Retriever:
    """In-memory BM25 Okapi lexical retriever."""

    def __init__(self, corpus: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self.num_docs = len(corpus)
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower().strip(".,!?:;\"'()[]{}") for w in text.split() if w.strip(".,!?:;\"'()[]{}")]

    def _build_index(self):
        total_len = 0
        for doc in self.corpus:
            doc_id = doc["id"]
            tokens = self._tokenize(doc["content"])
            self.doc_len[doc_id] = len(tokens)
            total_len += len(tokens)
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs[doc_id] = tf
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        self.avg_doc_len = total_len / self.num_docs if self.num_docs > 0 else 1.0

    def score(self, query: str) -> List[Tuple[str, float]]:
        """
        TODO 1: Return [(doc_id, bm25_score), ...] for EVERY document, sorted best first.

        Formula:
          IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
          Score(D, Q) = sum over query terms t of
              IDF(t) * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (|D| / avgdl)))

        Hints: use self.doc_freqs, self.term_freqs, self.doc_len, self.avg_doc_len.
               Guard against division by zero. Skip query terms that are not in the index.
        """
        query_tokens = self._tokenize(query)
        scores: List[Tuple[str, float]] = []
        for doc in self.corpus:
            scores.append((doc["id"], 0.0))   # placeholder: replace with real BM25 scores
        return scores


# ==============================================================================
# Part 3: Reciprocal Rank Fusion (RRF)
# ==============================================================================

def reciprocal_rank_fusion(
    dense_ranked_ids: List[str],
    sparse_ranked_ids: List[str],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    TODO 2: Merge two ranked lists of doc IDs. Return [(doc_id, rrf_score), ...] best first.

        RRF_Score(d) = sum over lists m of  1 / (k + rank_m(d)),   rank starts at 1

    A document missing from one list simply gets no contribution from that list.
    Check yourself with the slide 8 table: A = 0.0323, B = 0.0325, C = 0.0159.
    """
    scores: Dict[str, float] = {}
    for doc_id in dense_ranked_ids + sparse_ranked_ids:
        scores[doc_id] = 0.0                 # placeholder: replace with RRF
    return list(scores.items())


# ==============================================================================
# Part 4: Cross-Encoder Reranking Filter
# ==============================================================================

STOPWORDS = {"a", "an", "the", "of", "in", "on", "to", "and", "or", "with", "how", "does", "do", "is", "are", "for", "by"}


def _norm_tokens(text: str) -> set:
    """Lowercase, remove punctuation, drop stopwords, crude plural/verb-ending stemming."""
    out = set()
    for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()):
        if w in STOPWORDS:
            continue
        for suf in ("es", "s"):
            if w.endswith(suf) and len(w) > 4:
                w = w[: -len(suf)]
                break
        if w.endswith("e") and len(w) > 4:
            w = w[:-1]
        out.add(w)
    return out


def cross_encoder_rerank(
    query: str,
    candidate_docs: List[Dict[str, str]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    TODO 3: Score each candidate AGAINST THE QUERY and return the best top_k as dicts with keys
            id, title, content, rerank_score (rounded to 4 decimals), sorted best first.

    A real cross-encoder reads query + passage together. In this lab we imitate it with a
    token-overlap score: overlap / number of query tokens (use _norm_tokens on both sides),
    plus a 0.1 bonus when the whole query string appears inside the passage.
    """
    scored: List[Dict[str, Any]] = []
    for doc in candidate_docs:
        scored.append({"id": doc["id"], "title": doc.get("title", ""), "content": doc["content"],
                       "rerank_score": 0.0})   # placeholder: compute a real score
    return scored[:top_k]


# ==============================================================================
# Part 5: Knowledge Triple Extractor
# ==============================================================================

RELATIONS = {" mitigates ": "mitigates", " relies on ": "relies_on", " combines ": "combines"}


def extract_triples_rule_based(document: Dict[str, str]) -> ExtractionResult:
    """
    TODO 4: Split the document into sentences and extract (subject, predicate, object) triples
            as EntityTriple objects. Use the fixed relation list in RELATIONS (the "schema"):
            if a sentence contains " mitigates ", " relies on " or " combines ", split on it.
            Strip whitespace; use the predicate name from RELATIONS (e.g. "relies_on").
    (Optional extension: replace the rules with an LLM + Pydantic structured output, as in Week 1.)
    """
    triples: List[EntityTriple] = []
    return ExtractionResult(document_id=document["id"], triples=triples)


# ==============================================================================
# Provided helpers (do not edit)
# ==============================================================================

# A toy "embedding model": each dimension is a CONCEPT, so synonyms land on the same axis.
# Unknown tokens (such as ERR-4039 or XJ-4021) are ignored - which mimics dense retrieval's
# weakness on exact identifiers. A real system would call an embedding model instead.
CONCEPTS = [
    {"car", "cars", "automobile", "vehicle"},
    {"repair", "repairs", "maintenance", "fix", "servicing", "service"},
    {"search", "retrieval", "retrieve", "lookup", "query", "queries"},
    {"hybrid", "fusion", "combine", "combines", "blend"},
    {"embedding", "embeddings", "vector", "vectors", "semantic", "dense"},
    {"keyword", "keywords", "lexical", "bm25", "sparse", "exact", "identifiers"},
    {"rerank", "reranking", "reranker", "cross-encoder", "bi-encoder"},
    {"hallucination", "hallucinations", "poisoning", "noise", "irrelevant", "failure", "failures", "mismatch"},
    {"graph", "graphrag", "knowledge", "triples", "triple", "multi-hop", "entity"},
    {"summarize", "summarization", "summary", "themes", "corpus", "corpora"},
    {"chunk", "chunks", "chunking", "windowing", "split", "splits", "paragraph", "paragraphs"},
    {"database", "timeout", "connection", "error"},
]


def toy_embed(text: str) -> np.ndarray:
    vec = np.zeros(len(CONCEPTS))
    for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()):
        for i, group in enumerate(CONCEPTS):
            if w in group:
                vec[i] += 1.0
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def dense_rank(query: str, corpus: List[Dict[str, str]]) -> List[str]:
    q = toy_embed(query)
    scored = [(d["id"], cosine_similarity(q, toy_embed(d["content"]))) for d in corpus]
    scored.sort(key=lambda x: x[1], reverse=True)   # stable: ties keep corpus order
    return [doc_id for doc_id, s in scored if s > 0]   # zero similarity = not retrieved


def build_graph(results: List[ExtractionResult]) -> nx.DiGraph:
    G = nx.DiGraph()
    for result in results:
        for triple in result.triples:
            G.add_edge(triple.subject, triple.object, relation=triple.predicate)
    return G


def run_query(query: str, corpus: List[Dict[str, str]], bm25: BM25Retriever) -> None:
    print(f"Query: '{query}'")
    sparse_ids = [doc_id for doc_id, s in bm25.score(query) if s > 0]   # zero score = not retrieved
    dense_ids = dense_rank(query, corpus)
    print(f"  BM25  Top-3: {sparse_ids[:3]}")
    print(f"  Dense Top-3: {dense_ids[:3]}")
    fused = reciprocal_rank_fusion(dense_ids, sparse_ids, k=60)
    print("  RRF merged:  " + ", ".join(f"{d} ({s:.5f})" for d, s in fused[:3]))
    keep = {d for d, _ in fused[:5]}
    final = cross_encoder_rerank(query, [d for d in corpus if d["id"] in keep], top_k=2)
    for p in final:
        print(f"  Rerank [{p['id']}] score={p['rerank_score']}: {p['content'][:70]}...")
    print()


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_corpus.json")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please ensure sample_corpus.json exists.")
        return
    with open(data_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} documents.\n")

    bm25 = BM25Retriever(corpus)
    run_query("How does hybrid search mitigate exact keyword failure in naive vector RAG?", corpus, bm25)
    run_query("ERR-4039", corpus, bm25)          # exact identifier: BM25 should win
    run_query("car repair tips", corpus, bm25)   # paraphrase: dense should win

    print("Extracting knowledge triples and building a graph...")
    results = [extract_triples_rule_based(d) for d in corpus]
    for r in results:
        for t in r.triples:
            print(f"  [{r.document_id}] ({t.subject}) --[{t.predicate}]--> ({t.object})")
    G = build_graph(results)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    start = "Reciprocal rank fusion"
    if start in G:
        print(f"Two-hop neighborhood of '{start}': {sorted(nx.single_source_shortest_path_length(G, start, cutoff=2))}")


if __name__ == "__main__":
    main()
