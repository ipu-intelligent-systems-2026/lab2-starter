# Lab 2: Grounding & Context Engineering (RAG to GraphRAG)
# 演習2:グラウンディングと文脈工学(RAGからGraphRAGへ)

**Course / 科目:** Intelligent System Development (知能システム開発特論 2026)  
**Affiliation / 所属:** Iwate Prefectural University (IPU) / 岩手県立大学  
**Organization / 組織:** `ipu-intelligent-systems-2026`

---

## Objectives / 目的

**English:** Build beyond naive vector lookups. You will complete four TODOs in `lab2_starter.py`:

**日本語:** 単純なベクトル検索を超えるシステムを作ります。`lab2_starter.py` の4つのTODOを完成させてください。

| TODO | What you implement / 実装する内容 | Slide / スライド |
|------|-----------------------------------|------------------|
| 1 | BM25 Okapi scoring (exact keywords, part numbers, error codes) / BM25 Okapi スコア(完全一致キーワード、部品番号、エラーコード) | 7 |
| 2 | Reciprocal Rank Fusion (RRF) of dense + sparse rankings / 密検索と疎検索の順位を統合する RRF | 8 |
| 3 | Cross-encoder style reranker (100 candidates → best 3) / クロスエンコーダ型リランカー(100件の候補 → 上位3件) | 9 |
| 4 | Knowledge triple extraction (Subject, Predicate, Object) with Pydantic, then a NetworkX graph / Pydantic による知識トリプル(主語、述語、目的語)の抽出と、NetworkX のグラフ | 11 |

The toy dense retriever, `build_graph`, and `main()` are already provided. Do not edit them.  
簡易の密検索、`build_graph`、`main()` は用意済みです。編集しないでください。

## Files / ファイル

- `lab2_starter.py` — your code (edit the four TODO blocks only) / あなたが書くコード(4つのTODOだけを編集)
- `sample_corpus.json` — 10 short documents, including error codes (`ERR-4039`), a part number (`XJ-4021`) and a paraphrase example (automobile maintenance) / 10件の短い文書。エラーコード(`ERR-4039`)、部品番号(`XJ-4021`)、言い換えの例(自動車整備)を含む
- `lab2-starter.ipynb` — the same lab as a notebook, for Google Colab or Jupyter (self-contained, 5 documents) / 同じ演習のノートブック版。Google Colab や Jupyter 用(単独で動作、文書5件)
- `tests/test_lab2.py` — the automated checks / 自動テスト
- `requirements.txt`, `pytest.ini`

## Setup / 環境構築

**English:** Create a virtual environment and install the libraries.  
**日本語:** 仮想環境を作成し、ライブラリをインストールします。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Choose how to work / 作業方法を選ぶ

**English:** You can do the lab in a Python file (Option A) or in a notebook (Option B). Both have the same four TODOs.

**日本語:** 演習は、Python ファイル(方法A)でもノートブック(方法B)でもできます。TODOはどちらも同じ4つです。

### Option A: `lab2_starter.py` (local computer) / 方法A:`lab2_starter.py`(自分のパソコン)

1. **English:** Finish the Setup above, then open `lab2_starter.py` in your editor (for example VS Code).  
   **日本語:** 上の環境構築を終えてから、エディタ(VS Code など)で `lab2_starter.py` を開きます。
2. **English:** Find the four `TODO` blocks. Replace the placeholder lines with your own code. Do not edit the provided helper functions or `main()`.  
   **日本語:** `TODO` と書かれた4か所を探し、仮の行を自分のコードに書き換えます。用意済みの補助関数と `main()` は編集しないでください。
3. **English:** Run the program and read the output:  
   **日本語:** プログラムを実行して、出力を確認します。
   ```bash
   python lab2_starter.py
   ```
   **English:** The program runs from the start, but the TODO functions return placeholder values (zero scores, empty triples). When you finish, you should see:
   - Query `ERR-4039`: BM25 finds `doc_006`, while the dense list finds nothing (exact IDs are its weakness).
   - Query `car repair tips`: the dense list finds `doc_005`, while BM25 finds nothing (paraphrase is BM25's weakness).
   - RRF keeps the strengths of both, and the triples form a small graph.

   **日本語:** プログラムは最初から動きますが、TODOの関数はまだ仮の値(スコア0、トリプルなし)を返します。完成すると、次のようになります。
   - 検索 `ERR-4039`:BM25 は `doc_006` を見つけますが、密検索は何も見つけません(完全一致のIDは密検索の弱点です)。
   - 検索 `car repair tips`:密検索は `doc_005` を見つけますが、BM25 は何も見つけません(言い換えは BM25 の弱点です)。
   - RRF は両方の長所を残し、トリプルは小さなグラフになります。
4. **English:** Check your work with `pytest` (see Verify below).  
   **日本語:** `pytest` で確認します(下の「確認」を参照)。

### Option B: `lab2-starter.ipynb` (Google Colab or Jupyter) / 方法B:`lab2-starter.ipynb`(Google Colab または Jupyter)

**Google Colab (no installation / インストール不要)**

1. **English:** Open <https://colab.research.google.com>, choose **File → Upload notebook**, and select `lab2-starter.ipynb`.  
   **日本語:** <https://colab.research.google.com> を開き、**ファイル → ノートブックをアップロード** で `lab2-starter.ipynb` を選びます。
2. **English:** Run the cells from top to bottom (**Shift + Enter**). The first code cell installs the libraries and takes about a minute.  
   **日本語:** セルを上から順に実行します(**Shift + Enter**)。最初のコードセルがライブラリをインストールします(1分ほどかかります)。
3. **English:** At each TODO cell, replace the placeholder lines with your code and run the cell. Then run the test cell right below it. It prints `passed` or tells you what is still missing, and it does not stop the notebook.  
   **日本語:** 各TODOのセルで、仮の行を自分のコードに書き換えて、セルを実行します。次に、すぐ下のテスト用セルを実行します。「合格」か、まだ足りない点が表示されます。ノートブックは止まりません。
4. **English:** Run the last two cells to see the full pipeline and the knowledge graph.  
   **日本語:** 最後の2つのセルを実行して、パイプライン全体と知識グラフを確認します。

**Jupyter on your computer / 自分のパソコンの Jupyter**
```bash
pip install notebook
jupyter notebook lab2-starter.ipynb
```
**English:** In VS Code you can also open the `.ipynb` file directly and choose your `.venv` as the kernel.  
**日本語:** VS Code では、`.ipynb` ファイルをそのまま開き、カーネルとして `.venv` を選ぶこともできます。

> **English — Important:** The notebook has its own small corpus (5 documents: `doc_01` to `doc_05`), so the document IDs differ from Option A (for example `ERR-4039` finds `doc_03` here). The automated tests and the GitHub check run on `lab2_starter.py`, **not** on the notebook. If you worked in the notebook, copy your four finished TODO functions into `lab2_starter.py`, run `pytest`, and then submit.
>
> **日本語 — 重要:** ノートブックには独自の小さなコーパス(文書5件:`doc_01`〜`doc_05`)があるため、文書IDが方法Aと異なります(たとえば `ERR-4039` はここでは `doc_03` が見つかります)。自動テストと GitHub のチェックは、ノートブックではなく `lab2_starter.py` に対して実行されます。ノートブックで作業した場合は、完成した4つのTODO関数を `lab2_starter.py` にコピーして、`pytest` を実行してから提出してください。

## Verify / 確認

```bash
pytest
```

**English:** Work through the failing tests one TODO at a time. The RRF tests reproduce the table from Slide 8 (A = 0.0323, B = 0.0325, C = 0.0159).

**日本語:** 失敗しているテストを、TODOを1つずつ直しながら確認してください。RRF のテストは、スライド8の表(A = 0.0323、B = 0.0325、C = 0.0159)を再現します。

## Submission / 提出

```bash
git add .
git commit -m "Complete Lab 2 hybrid retrieval and triple extraction"
git push origin main
```

**English:** Each push runs `pytest` on `lab2_starter.py` through GitHub Actions. Your push is your official submission for Week 2. If you also changed the notebook, push it too (`git add .` includes it).

**日本語:** プッシュするたびに、GitHub Actions で `lab2_starter.py` に対して `pytest` が実行されます。このプッシュが、第2回の正式な提出になります。ノートブックも変更した場合は、それもプッシュしてください(`git add .` に含まれます)。
