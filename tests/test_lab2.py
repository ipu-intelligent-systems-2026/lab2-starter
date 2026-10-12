"""Autograder for Lab 2. Run with: pytest   (instructors: LAB_MODULE=lab2_solution pytest)"""
import importlib, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = importlib.import_module(os.environ.get("LAB_MODULE", "lab2_starter"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = json.load(open(os.path.join(ROOT, "sample_corpus.json"), encoding="utf-8"))


# ---- TODO 1: BM25 -----------------------------------------------------------
def test_bm25_exact_identifier_ranks_first():
    top = L.BM25Retriever(CORPUS).score("ERR-4039")[0]
    assert top[0] == "doc_006" and top[1] > 0

def test_bm25_returns_every_document_sorted():
    res = L.BM25Retriever(CORPUS).score("hybrid search keyword")
    assert len(res) == len(CORPUS)
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)

def test_bm25_matches_formula():
    docs = [{"id": "a", "content": "apple banana"}, {"id": "b", "content": "banana cherry cherry"}]
    r = L.BM25Retriever(docs)
    got = dict(r.score("apple"))["a"]
    idf = math.log((2 - 1 + 0.5) / (1 + 0.5) + 1)
    denom = 1 + 1.5 * (1 - 0.75 + 0.75 * (2 / 2.5))
    assert abs(got - idf * (1 * 2.5) / denom) < 1e-9

def test_bm25_empty_document_does_not_crash():
    docs = [{"id": "a", "content": ""}, {"id": "b", "content": "hello world"}]
    assert dict(L.BM25Retriever(docs).score("hello"))["a"] == 0.0


# ---- TODO 2: RRF ------------------------------------------------------------
def test_rrf_rank_is_one_indexed():
    assert abs(dict(L.reciprocal_rank_fusion(["a"], []))["a"] - 1 / 61) < 1e-12

def test_rrf_slide8_example():
    res = dict(L.reciprocal_rank_fusion(["A", "B", "C"], ["B", "X", "A"]))
    assert abs(res["A"] - (1 / 61 + 1 / 63)) < 1e-9      # 0.0323
    assert abs(res["B"] - (1 / 62 + 1 / 61)) < 1e-9      # 0.0325
    assert abs(res["C"] - 1 / 63) < 1e-9                 # 0.0159

def test_rrf_sorted_best_first():
    ids = [d for d, _ in L.reciprocal_rank_fusion(["A", "B", "C"], ["B", "X", "A"])]
    assert ids[:2] == ["B", "A"]


# ---- TODO 3: reranker -------------------------------------------------------
def test_rerank_top_k_and_order():
    q = "hybrid search keyword failure"
    out = L.cross_encoder_rerank(q, CORPUS, top_k=3)
    assert len(out) == 3
    assert out[0]["id"] == "doc_001"
    assert [o["rerank_score"] for o in out] == sorted((o["rerank_score"] for o in out), reverse=True)
    assert out[0]["rerank_score"] > 0

def test_rerank_ignores_punctuation():
    docs = [{"id": "x", "content": "keyword failure."}, {"id": "y", "content": "unrelated text"}]
    out = L.cross_encoder_rerank("keyword failure?", docs, top_k=2)
    assert out[0]["id"] == "x" and out[0]["rerank_score"] >= 1.0


# ---- TODO 4: triples + graph ------------------------------------------------
def _doc(i):
    return next(d for d in CORPUS if d["id"] == i)

def test_triples_mitigates():
    r = L.extract_triples_rule_based(_doc("doc_001"))
    assert any(t.subject == "Reciprocal rank fusion" and t.predicate == "mitigates"
               and t.object == "ranking distribution differences" for t in r.triples)

def test_triples_relies_on_uses_schema_name():
    r = L.extract_triples_rule_based(_doc("doc_003"))
    assert any(t.predicate == "relies_on" and t.subject == "GraphRAG" for t in r.triples)

def test_graph_two_hop_walk():
    import networkx as nx
    r = [L.extract_triples_rule_based(d) for d in CORPUS]
    G = L.build_graph(r)
    assert G.number_of_edges() >= 4
    assert "Reciprocal rank fusion" in G
