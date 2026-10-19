# RAGForge

RAGForge is a hands-on project that builds a Retrieval-Augmented Generation (RAG) system from first principles. The long-term goal is a RAG system that is **evaluation-driven, self-correcting, and attack-resistant**:

- answers grounded in retrieved evidence, with source citations;
- detection and correction of claims the evidence does not support;
- defenses against prompt-injection content hidden in documents;
- every design choice justified by measured retrieval and answer quality.

The project deliberately avoids RAG frameworks (LangChain, LlamaIndex) and vector databases so that every step can be inspected and explained.

**Current status:**

- **M1** (document ingestion and dense retrieval): complete.
- **M1.5** (evidence-level retrieval benchmark): complete.
- **M2** (BM25, hybrid retrieval with Reciprocal Rank Fusion, method comparison): complete.
- **M2.5** (weighted RRF, four fixed weightings compared on the development split): complete.
- M3 and later: not implemented.

## M1 Architecture

```
Documents            data/documents/*.json (24 synthetic policies)
   ↓
Ingestion            src/ingest.py      load + validate metadata, unique IDs, non-empty sections
   ↓
Chunking             src/chunking.py    section-aware, ≤120 words, sentence boundaries
   ↓
Embedding Model      src/embeddings.py  all-MiniLM-L6-v2 → unit-normalized 384-d vectors
   ↓
NumPy Cosine Similarity  src/retrieve.py  scores = chunk_matrix @ query_vector
   ↓
Top-K Retrieval      src/retrieve.py    stable sort, return the 5 best chunks
```

The corpus is embedded once at startup. Each question then costs one model forward pass plus one matrix-vector product.

M2 adds a second, lexical retriever over the same 92 chunks and a fusion step:

```
                 ┌─ Dense (MiniLM, cosine) ── full ranking of 92 chunks ─┐
Question ────────┤                                                       ├─ Reciprocal Rank Fusion ─→ Top-K
                 └─ BM25 (src/bm25.py) ─── chunks sharing ≥1 term ───────┘   (src/fusion.py, k = 60)
```

### Repository layout

```
data/documents/                      24 policy documents, one JSON file each
data/evaluations/retrieval_questions.json   55 labeled questions (section labels, dev/test split)
src/ingest.py                        loading and validation
src/chunking.py                      section-aware chunking
src/embeddings.py                    Sentence Transformers wrapper + normalization
src/bm25.py                          tokenizer and BM25 index/scoring, from scratch
src/fusion.py                        Reciprocal Rank Fusion
src/retrieve.py                      dense retrieval, search() for dense/bm25/hybrid, retrieval CLI
src/evaluate.py                      benchmark: metrics, splits, failure analysis, method comparison
tests/test_retrieval.py              M1 / M1.5 tests
tests/test_hybrid.py                 M2 tests (BM25, RRF, retrieval interface)
tests/fixtures/dense_regression.json dense top-5 results recorded before M2, for regression checks
```

### The synthetic knowledge base

There are 24 fictional policies, six in each of four departments:

| Department | Documents |
|---|---|
| Human Resources | HR-101 Annual Leave, HR-102 Remote Work, HR-103 Onboarding, HR-104 Parental Leave, HR-105 Professional Development, HR-106 Employee Expense Reimbursement |
| Finance | FIN-201 Travel Expenses, FIN-202 Procurement Approval, FIN-203 Invoice Payments, FIN-204 Corporate Credit Cards, FIN-205 Budget Approvals, FIN-206 Vendor Payments |
| Security | SEC-301 Passwords, SEC-302 MFA, SEC-303 Data Classification, SEC-304 Incident Reporting, SEC-305 Device Security, SEC-306 Third-Party Access |
| Customer Policies | REF-728 Domestic Refunds, REF-729 International Refunds, SUB-730 Subscription Cancellation, WAR-731 Warranty Claims, SHP-732 Shipping and Delivery, ESC-733 Support Escalation |

Each document has an ID, title, department, effective date, and three or four sections. Every section has its own ID, such as `REF-729.2`. The policies are designed to be easy to confuse:

- near-duplicate pairs with different rules, such as domestic refunds (30 days, free return label) and international refunds (45 days, customer pays return shipping);
- shared vocabulary such as "receipt", "approval", "within N business days", "report within 1 hour" and "Finance Director";
- specific thresholds, deadlines, exceptions, and cross-references between policies.

The documents are internally consistent. They contain no malicious or conflicting content; that belongs to later milestones.

## M1 Learning Objectives

**What embeddings are.** An embedding model maps text to a fixed-length vector. The text is first tokenized into subword IDs. A transformer encoder then turns those IDs into contextual token vectors, and pooling averages the token vectors into one vector per passage. The model is trained so that passages with similar meaning point in similar directions. A question about "overseas customers" can therefore match a passage about "international customers" even though the words differ. all-MiniLM-L6-v2 produces 384-dimensional vectors. The code reads the dimension from the model's actual output rather than assuming it.

**Why chunking is necessary.** One vector for a whole document blurs many topics together, and the model truncates inputs longer than 256 tokens. Splitting each document into focused passages gives sharper vectors and lets retrieval point at the rule that answers the question. RAGForge chunks by section, because the documents already have meaningful sections. It splits a section only when it exceeds 120 words, and then only at sentence boundaries. Chunks never mix text from different sections or documents.

Each chunk is embedded as `"{title} - {section_title} ({section_id}): {text}"`. A chunk such as "Requests must be submitted at least 30 days before departure" then still carries the context that it belongs to the Remote Work Policy. This design choice was made before the evaluation was run, not tuned against it.

**How cosine similarity works.** Cosine similarity measures the angle between two vectors:

```
cosine(a, b) = (a · b) / (‖a‖ ‖b‖)
```

It is 1 for vectors pointing the same way, 0 for orthogonal vectors, and -1 for opposite vectors, and it ignores vector length. After every embedding is scaled to unit length, the denominator is 1. The similarities between one query and all N chunks then become a single matrix-vector product, `chunk_matrix @ query`, with shape (N, 384) @ (384,) giving (N,).

**What top-K retrieval means.** After scoring every chunk, the K highest-scoring chunks are returned in descending order (K = 5 by default). These K passages are all that a later generation step would ever see.

**Why retrieval quality matters before generation.** A language model can only ground its answer in what retrieval gives it. If the right passage is missing from the top K, the best case is "I don't know". The worst case is a fluent answer built from the wrong policy, such as quoting the domestic 30-day refund window to an international customer. Retrieval is also cheap to measure, so it is the first component to get right and to evaluate.

## Setup

These steps assume Python 3.11–3.13. M1 was developed and tested on Python 3.13.3 with numpy 2.5.4, torch 2.14.1 and sentence-transformers 6.1.0.

1. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   The first run downloads all-MiniLM-L6-v2 (about 90 MB) to the standard Hugging Face cache (`~/.cache/huggingface`). Model weights are never stored in the repository.
3. **Run the retrieval CLI**
   ```bash
   python -m src.retrieve                    # interactive, dense; type 'exit' or 'quit' to stop
   python -m src.retrieve --question "What is the deadline for returning an overseas order?"
   python -m src.retrieve --top-k 3          # change the number of results

   python -m src.retrieve --question "What does FIN-205.3 say?" --method dense
   python -m src.retrieve --question "What does FIN-205.3 say?" --method bm25     # no model download needed
   python -m src.retrieve --question "What does FIN-205.3 say?" --method hybrid   # shows RRF score + both ranks
   python -m src.retrieve --question "What does FIN-205.3 say?" --method hybrid --dense-weight 0.7 --bm25-weight 0.3
   ```
   `--method` defaults to `dense`, so the M1 commands behave as before. Hybrid weights default to 0.5 / 0.5, which ranks exactly like M2.
4. **Run the tests**
   ```bash
   python -m unittest discover -s tests -v                          # fast: no model needed
   RAGFORGE_MODEL_TESTS=1 python -m unittest discover -s tests -v   # also loads the embedding model
   ```
5. **Run the retrieval benchmark** (see [M1.5](#m15--retrieval-evaluation))
   ```bash
   python -m src.evaluate --split development              # development split, dense (defaults)
   python -m src.evaluate --split development --method bm25          # one method: dense | bm25 | hybrid
   python -m src.evaluate --split development --failures   # plus per-question results and failure analysis
   python -m src.evaluate --split development --compare    # dense, BM25 and four RRF weightings, with latency
   python -m src.evaluate --split development --compare --failures   # plus per-question changes and section ranks
   python -m src.evaluate --split test --compare --confirm-held-out  # held-out split: final reporting only
   python -m src.evaluate --split all --confirm-held-out             # all 55 questions: descriptive only
   ```
   Since M2.5, the test and all splits refuse to run without `--confirm-held-out`, so held-out questions are never evaluated by accident.
   `python -m src.retrieve --evaluate [--method ...]` is kept as an alias for the development-split command.

## M1 Baseline Evaluation (historical)

M1 shipped with 29 hand-labeled questions in four categories: exact keyword, semantic paraphrase, exact policy identifier (such as "What does FIN-205.3 say?"), and similar-policy disambiguation (such as domestic versus international refunds).

For each question, the top 5 **chunks** are retrieved and collapsed into a ranked list of unique **document** IDs (the first occurrence keeps its rank):

- **Recall@5** is |relevant ∩ retrieved documents| / |relevant|, averaged over questions.
- **MRR@5** is 1 / (rank of the first relevant document in that list), or 0 if none appears, averaged over questions.

These were the M1 results over the original 29 questions, with all-MiniLM-L6-v2, 92 chunks and a 120-word limit. M1.5 replaced this evaluation; see below.

| Category | N | Recall@5 | MRR@5 |
|---|---|---|---|
| exact_identifier | 7 | 1.000 | 0.643 |
| exact_keyword | 7 | 1.000 | 1.000 |
| semantic_paraphrase | 7 | 1.000 | 1.000 |
| similar_policy_disambiguation | 8 | 1.000 | 1.000 |
| **overall** | **29** | **1.000** | **0.914** |

**This is an initial development baseline, not a statistically reliable benchmark.** With 29 questions, a single question changes a category score by about 0.14. Some observations:

- Document-level Recall@5 is saturated, so this set cannot yet tell good retrievers from better ones. Harder questions, chunk-level metrics, or Recall@1 will be needed to compare methods in M2.
- Exact-identifier questions are the weak spot. Questions such as "What does FIN-205.3 say?" find the right document, but often at rank 2 behind a document with similar wording. Dense embeddings capture meaning, not exact identifiers, so this is the case where adding BM25 lexical matching in M2 should help.

## M1.5 — Retrieval Evaluation

M1's document-level benchmark scored Recall@5 = 1.000, so it could not tell a good retriever from a better one. M1.5 measures whether retrieval finds the **evidence**, meaning the specific section that answers the question. The retriever, documents, chunking and embeddings are unchanged from M1. This milestone is measurement only.

### Why document-level evaluation can be misleading

A policy document has 3–4 sections on different topics. If the question asks about international store credit (REF-729.3) and retrieval returns REF-729.1 (refund deadlines), the document-level score is perfect, but a generator would see the wrong rule. With 24 documents and five chunks retrieved, the right document is almost always somewhere in the top five. Document-level Recall@5 is 1.000 on the development split, while the evidence section is ranked first for only 81% of questions.

### Ground truth

Every question has `relevant_document_ids` and `relevant_section_ids`. A section is labeled when its text states the rule needed for the answer. Sections that only mention the rule in passing, such as REF-729.4 saying "after the 45-day refund window", are not labeled. When an answer needs several rules, every section that holds one is labeled. The loader rejects any label that does not exist, any section whose document is not labeled, and any labeled document without a labeled section.

`python -m src.evaluate` checks that every label refers to a real section. It does not check that the section contains the answer. I checked that by hand when writing the labels and by matching an answer phrase against each labeled section's text. That script is not part of the repository.

| Category | What it tests | Total | Dev | Test |
|---|---|---|---|---|
| exact_keyword | question reuses the policy's own terms | 7 | 5 | 2 |
| exact_identifier | question cites a section ID such as `FIN-205.3` | 10 | 7 | 3 |
| semantic_paraphrase | different vocabulary from the source | 11 | 8 | 3 |
| similar_policy_disambiguation | related policies with different rules | 12 | 8 | 4 |
| numerical_threshold | exact amounts, durations, deadlines | 5 | 3 | 2 |
| policy_exception | the exception, not the general rule | 5 | 3 | 2 |
| multi_section | needs two or three sections | 5 | 3 | 2 |
| **total** | | **55** | **37** | **18** |

Questions Q001–Q029 are the original M1 questions, unchanged apart from the added section labels and split. Questions Q030–Q055 were written and labeled in one pass before the retriever was run on them, and none were edited or removed after seeing results.

### Metrics

All metrics use the top 5 chunks the retriever returns and are averaged over questions.

| Metric | Ranked list it uses | Per-question value |
|---|---|---|
| Document Recall@5 | chunks collapsed to unique documents (M1 convention) | share of relevant documents found |
| Section Recall@1 | section of chunk 1 | share of relevant sections found at rank 1 |
| Section Hit@1 | section of chunk 1 | 1 if chunk 1 is from any relevant section |
| Section Recall@5 | sections of chunks 1–5 | share of unique relevant sections found |
| Section MRR@5 | sections of chunks 1–5 | 1 / chunk rank of first relevant section, else 0 |

- **Recall@K vs MRR@K.** Recall@K asks whether the evidence is in the top K at all: did the generator get to see it? MRR@K asks how high the first piece of evidence is: is it near the top or buried at rank 5? Two retrievers can both have Recall@5 = 1 while one ranks the evidence first and the other fifth. MRR separates them.
- **Section Recall@1 vs Hit@1.** For a question with one relevant section they are identical. For a question with three relevant sections, a correct first chunk gives Hit@1 = 1 but Recall@1 = 1/3, because one chunk can hold only one section. Hit@1 asks whether the top result is useful. Recall@1 asks how much of the evidence the top result covers.
- **How multi-section questions are scored.** They get partial credit: finding 2 of 3 required sections in the top 5 gives Section Recall@5 = 0.667. Sections are counted once, so a section that was split into two chunks cannot be counted twice. MRR uses the rank of the first relevant chunk only.
- **Section ranks are chunk ranks.** Section metrics use the raw chunk list without collapsing it, so a relevant section at chunk 3 has reciprocal rank 1/3. Document metrics keep the M1 convention of collapsing to unique documents first. The two levels are therefore not directly comparable.

### Development and held-out test splits

Each question has `split: development` or `split: test`. The split is deterministic and stratified by category: `assign_splits()` in `src/evaluate.py` sorts each category's IDs, shuffles them with a fixed seed (1729), and puts the first `round(0.3 × n)` into test. The recipe uses only IDs and categories, never retrieval results. A unit test checks that the stored splits match it.

Rules for M2 and later:

1. Inspect failures, compare approaches and choose parameters (such as BM25 settings, fusion weights or top-K) **only on the development split**. `--failures` works only for that split.
2. Freeze the configuration, then run it on the test split once and report development and test scores separately.
3. Never change documents, labels or retrieval settings in response to test results.

This is a **held-out query split, not a held-out corpus**. Test questions ask about the same 24 documents (and sometimes the same sections) as development questions. It estimates how a frozen configuration handles new questions about known documents. It says nothing about how it would handle unseen documents, departments or writing styles.

### Results: M1 dense retriever on the development split

From `python -m src.evaluate --split development`, all-MiniLM-L6-v2 with 92 chunks:

| Metric | Development (37) |
|---|---|
| Document Recall@5 | 1.000 |
| Section Recall@1 | 0.752 |
| Section Hit@1 | 0.811 |
| Section Recall@5 | 0.964 |
| Section MRR@5 | 0.887 |

| Category (dev) | N | Section Hit@1 | Section Recall@5 | Section MRR@5 |
|---|---|---|---|---|
| exact_keyword | 5 | 1.000 | 1.000 | 1.000 |
| exact_identifier | 7 | 0.429 | 0.857 | 0.643 |
| semantic_paraphrase | 8 | 0.875 | 1.000 | 0.917 |
| similar_policy_disambiguation | 8 | 0.750 | 1.000 | 0.875 |
| numerical_threshold | 3 | 1.000 | 1.000 | 1.000 |
| policy_exception | 3 | 1.000 | 1.000 | 1.000 |
| multi_section | 3 | 1.000 | 0.889 | 1.000 |

Categories with 3 questions move by 0.33 per question, so read their scores as anecdotes. The development failure analysis found 7 top-1 misses. Four cite a section identifier. One of these, "What does FIN-205.3 say?", ranks its evidence 62nd of 92 chunks. Two put the right document's wrong section first, and one confused the warranty policy with the shipping policy. Q051 also misses one of its three required sections.

### Limitations

- **Small synthetic corpus.** There are 24 short, clean, consistently formatted documents written by one author. Real corpora are larger and noisier: duplicates, outdated versions, tables, PDFs. Scores here will not transfer.
- **Small question set.** A single question moves a development category score by 0.12–0.33, so differences of a few points between methods are noise.
- **Sections are not answers.** A section label says where the evidence lives, not that the exact sentence was retrieved or that a generator would use it correctly. Answer-level evaluation comes later (M4–M5).
- **One labeler, who also wrote the documents.** Questions may unconsciously follow the documents' wording. Labels for questions that could be read more than one way are judgment calls.
- **Leakage risk.** Development and test questions share documents and sometimes sections, and the test split has been run once to verify the command. Any change made after looking at test numbers would undermine the split.

## M2 — BM25 and Hybrid Retrieval

M2 adds a lexical retriever (BM25) and a hybrid retriever that fuses dense and BM25 rankings with Reciprocal Rank Fusion (RRF). The dense retriever is unchanged. A model test compares its top-5 results and scores against [tests/fixtures/dense_regression.json](tests/fixtures/dense_regression.json), which was recorded before any M2 change. No parameter was tuned: BM25 uses k1 = 1.5 and b = 0.75, and RRF uses k = 60, all conventional defaults chosen before the first evaluation.

### What BM25 is

BM25 is the classic search-engine ranking function. It scores a chunk by the query words it contains:

```
BM25(q, d) = Σ_t∈q  IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
IDF(t)     = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))        (Lucene variant, always > 0)
```

- **Term frequency** `f(t,d)`: more occurrences mean a higher score, with diminishing returns controlled by **k1**. The score for one term can never exceed `IDF·(k1+1)`.
- **Document frequency** `df(t)` and **IDF**: a term found in few chunks (such as `fin-205.3`, df = 1 of 92) is worth far more than one found in many (such as `day`, df = 52).
- **Length normalization** with **b**: a match in a short chunk counts more than the same match in a long one. With b = 0 length is ignored; with b = 1 it is fully normalized.
- The sum runs over unique query terms, so repeating a word in the question does not boost it.

[src/bm25.py](src/bm25.py) implements this directly with `collections.Counter` and NumPy; no BM25 library is used. Each chunk is indexed using the same fields the dense model embeds: section ID, document title, section title and text.

**Tokenizer.** Text is lowercased.
- Policy identifiers stay whole: `FIN-205.3` becomes `fin-205.3`, plus `fin-205` so that a query naming the document also matches.
- Letter-digit tokens stay whole (`p1`, `fido2`).
- Numbers drop thousands separators, so `$25,000` matches `25000`.
- A small stopword list removes words like "what" and "the".
- A minimal plural stemmer maps `refunds` to `refund`.

The tradeoffs: there is no stemming beyond plurals (`approve` ≠ `approval`), no synonyms, and hyphenated words are split.

### Why dense retrieval struggles with exact identifiers

An embedding model represents meaning, and a string like "FIN-205.3" has very little of it. MiniLM's tokenizer splits it into the pieces `fin - 205 . 3`. The likely consequence (an explanation, not something measured here) is that these pieces barely distinguish FIN-202, FIN-205 and FIN-206, so meaning-based matching dominates. For "What does FIN-205.3 say?" the dense retriever ranks FIN-202.1 first and the correct section 62nd of 92. BM25 treats `fin-205.3` as a rare token that appears in exactly one chunk, so it ranks that chunk first.

### How hybrid retrieval and RRF work

Both retrievers rank the full corpus: dense ranks all 92 chunks, and BM25 ranks every chunk that shares at least one term with the question. The rankings are fused with

```
RRF(chunk) = Σ over rankings containing the chunk of  1 / (60 + rank)
```

Chunks are matched by `chunk_id`, and ties go to the better single rank and then to the chunk ID. The fusion runs on the full rankings rather than each method's top 5, so a relevant chunk that one method ranks low is not cut off before fusion.

**Why not add the raw scores?** Cosine similarities here are about 0.2–0.7, while BM25 scores range from 0 to about 30 and change scale with every query (more query terms give larger scores). A raw sum would be dominated by BM25 on some queries and barely affected by it on others. Ranks have the same meaning for both retrievers, so RRF needs no calibration. The cost is that RRF ignores how confident each retriever is.

**Worked example (actual values for "What does FIN-205.3 say?"):**

| Chunk | Dense rank | BM25 rank | RRF score | Hybrid rank |
|---|---|---|---|---|
| FIN-205.1 | 2 | 3 | 1/62 + 1/63 = 0.03200 | 1 |
| FIN-205.2 | 6 | 4 | 1/66 + 1/64 = 0.03078 | 2 |
| FIN-205.4 | 15 | 2 | 1/75 + 1/62 = 0.02946 | 3 |
| FIN-205.3 (the answer) | 62 | 1 | 1/122 + 1/61 = 0.02459 | 5 |

BM25 is very confident about FIN-205.3 (score 6.86 against 3.01 for the next chunk), but RRF sees only "rank 1". The dense rank of 62 then pulls it below sections that both retrievers ranked moderately well. Hybrid still moves the answer from 62nd to 5th, but not to 1st.

### Results on the development split (observed)

From `python -m src.evaluate --split development --compare`. Latency is query time only, from one run on an Apple Silicon laptop (CPU), so treat it as rough. One-time setup took 4.1 s to load the model, 0.18 s to embed the chunks and under 0.01 s to build the BM25 index.

| Metric (37 dev questions) | Dense | BM25 | Hybrid (RRF) |
|---|---|---|---|
| Document Recall@5 | **1.000** | 0.959 | 0.986 |
| Section Recall@1 | **0.752** | 0.631 | 0.712 |
| Section Hit@1 | **0.811** | 0.676 | 0.757 |
| Section Recall@5 | **0.964** | 0.901 | **0.964** |
| Section MRR@5 | **0.887** | 0.782 | 0.842 |
| Query latency, mean | 4.4 ms | 0.09 ms | 4.5 ms |

| Section Hit@1 by category (dev) | N | Dense | BM25 | Hybrid |
|---|---|---|---|---|
| exact_keyword | 5 | 1.000 | 1.000 | 1.000 |
| exact_identifier | 7 | 0.429 | **1.000** | 0.857 |
| semantic_paraphrase | 8 | **0.875** | 0.125 | 0.250 |
| similar_policy_disambiguation | 8 | 0.750 | 0.625 | **0.875** |
| numerical_threshold | 3 | **1.000** | 0.667 | 0.667 |
| policy_exception | 3 | **1.000** | 0.667 | **1.000** |
| multi_section | 3 | 1.000 | 1.000 | 1.000 |

**On this benchmark, hybrid retrieval with default RRF does not beat dense retrieval overall.** Dense missed 4 of the 7 exact-identifier questions at rank 1. Hybrid fixed 3 of them (Q015, Q016, Q021) and moved Q020 from rank 62 to rank 5. It also helped on Q036 (international store credit) and Q049 (warranty postage). It lost five semantic paraphrases and one numerical question that dense had right at rank 1. Of the 17 development questions where any method missed, hybrid helped on 6, hurt on 6 and made no difference on 5. BM25 alone is the best method for exact identifiers and the worst for paraphrases (Hit@1 0.125).

### Where BM25 or hybrid underperformed (observed)

- **Paraphrases with no shared words (Q010, Q013, Q047).** For "Someone took my work laptop out of my car", BM25 matches 20 chunks on words like "work" and "need", but not SEC-304.2 (how to report an incident). That chunk gets only its dense contribution, 1/64 = 0.0156, which is less than a chunk ranked 30th by both retrievers would get (2/90 = 0.0222). It falls from dense rank 4 to hybrid rank 21. Under RRF, a missing BM25 match acts as a penalty.
- **Near-misses that flip rank 1 (Q008, Q012, Q041).** Dense ranks the answer 1st and BM25 ranks it 2nd. The chunk BM25 prefers is also near the top of the dense list, so it wins the fusion by a small margin.
- **Exact identifier with a very low dense rank (Q020).** See the worked example above: hybrid improves the answer from rank 62 to 5, but it does not reach rank 1.
- **Multi-section coverage (Q051).** BM25 ranks FIN-202.2, the approval thresholds, 20th, because the question's "$30,000" does not appear in the section ("$5,000", "$50,000"). Hybrid recovers it only to rank 8.

These are observations about one corpus, one embedding model and default parameters. They are not general claims about BM25 or RRF. In theory, hybrid retrieval should help most when the two retrievers fail on different questions. Here they do fail on different questions, but plain RRF has no way to know which retriever to trust for a given question. Whether weighted fusion or reranking (M3) does better is an open question, to be tested on the development split.

### Held-out test split

M2 was compared on the development split only. Following the M1.5 protocol, the test split should be run once on frozen configurations (`python -m src.evaluate --split test --compare --confirm-held-out`) and reported next to the development numbers. That run has not been done yet.

### M2 limitations

- **Small synthetic corpus.** 24 clean documents with one consistent identifier scheme. In real corpora identifiers are inconsistent ("FIN 205.3", "section 3 of the budget policy"), and the vocabulary is far larger, which changes IDF.
- **Small development set.** A single question moves overall Hit@1 by 0.027, and a 3-question category by 0.33. The 0.05 gap between dense and hybrid is two questions.
- **The question mix decides the winner.** With 7 identifier questions and 8 paraphrases out of 37, the overall comparison mostly reflects that proportion. A corpus with more ID lookups would favor BM25 or hybrid.
- **Simple tokenizer.** Only plurals are stemmed, there are no synonyms, and the identifier pattern is fixed.
- **Untuned defaults.** k1, b and the RRF constant were not tuned, which keeps the comparison fair to the M1 baseline. Any tuning must use the development split only.
- **Latency** comes from a single CPU run on a 92-chunk corpus. The ordering (BM25 much faster than dense) is meaningful; the exact numbers are not.

## M2.5 — Weighted Reciprocal Rank Fusion

### Why equal-weight RRF underperformed dense retrieval

Equal RRF gives BM25 the same vote as dense retrieval on every question. On paraphrased questions, BM25's ranking mostly reflects incidental shared words like "work" or "need", yet it counts as much as the dense ranking. A chunk that BM25 does not match at all gets only half the possible score. In M2 this cost six questions that dense alone had right at rank 1, all of them paraphrases except one numerical question.

RRF also cannot use BM25's large score margins. It sees only ranks, so BM25 being far more confident about FIN-205.3 than about anything else (6.86 vs 3.01) counts no more than a narrow lead. Rank-only fusion is what makes RRF robust to the two incompatible score scales, and this is the price.

### What weighted RRF changes

```
WeightedRRF(d) = w_dense / (k + rank_dense(d)) + w_bm25 / (k + rank_bm25(d))      k = 60, ranks 1-based
```

A retriever that did not return the chunk contributes 0. The weights must be finite, between 0 and 1, sum to 1 (within 1e-9) and not both be zero; anything else is rejected with an error. Equal weights scale every score by the same constant, so 0.5 / 0.5 ranks exactly like M2's unweighted RRF, with every score halved. That is the default, so M2 hybrid results are unchanged, and a model test checks this against a fixture recorded before M2.5.

Only four configurations were evaluated, all fixed in advance, with no weight search and no change to k:

| Configuration | Dense weight | BM25 weight |
|---|---|---|
| Equal RRF (M2) | 0.5 | 0.5 |
| Weighted RRF A | 0.6 | 0.4 |
| Weighted RRF B | 0.7 | 0.3 |
| Weighted RRF C | 0.8 | 0.2 |

```bash
python -m src.evaluate --split development --compare              # all six configurations
python -m src.evaluate --split development --compare --failures   # plus per-question changes and section ranks
python -m src.retrieve --question "What does FIN-205.3 say?" --method hybrid --dense-weight 0.7 --bm25-weight 0.3
```

Each development question is retrieved once by dense and once by BM25, and all four fusions reuse those two rankings.

### Results on the development split (observed, 37 questions)

| Metric | Dense | BM25 | RRF .5/.5 | RRF .6/.4 | RRF .7/.3 | RRF .8/.2 |
|---|---|---|---|---|---|---|
| Document Recall@5 | 1.000 | 0.959 | 0.986 | 0.986 | 0.986 | 0.986 |
| Section Recall@1 | 0.752 | 0.631 | 0.712 | 0.793 | 0.766 | 0.793 |
| Section Hit@1 | 0.811 (30) | 0.676 (25) | 0.757 (28) | 0.838 (31) | 0.811 (30) | 0.838 (31) |
| Section Recall@5 | 0.964 | 0.901 | 0.964 | 0.977 | 0.950 | 0.950 |
| Section MRR@5 | 0.887 | 0.782 | 0.842 | 0.897 | 0.881 | 0.901 |
| Total query ms, mean | 4.31 | 0.09 | 4.56 | 4.51 | 4.51 | 4.52 |

The numbers in parentheses are questions with a relevant section at rank 1.

- **Latency** is split into three parts:
  - retrieval: dense, BM25, or both for hybrid, about 4.40 ms;
  - fusion only: 0.11–0.16 ms, which includes building result records for all 92 chunks;
  - total: retrieval plus fusion.

  It excludes model loading (5.2 s in this run), chunk embedding (0.36 s), building the BM25 index and one untimed warm-up query. It comes from a single CPU run.
- **Metrics:** the Dense, BM25 and RRF .5/.5 columns reproduce the M2 numbers exactly.

| Section Hit@1 by category | N | Dense | BM25 | .5/.5 | .6/.4 | .7/.3 | .8/.2 |
|---|---|---|---|---|---|---|---|
| exact_identifier | 7 | 0.429 | 1.000 | 0.857 | 0.857 | 0.857 | 0.857 |
| semantic_paraphrase | 8 | 0.875 | 0.125 | 0.250 | 0.500 | 0.500 | 0.625 |
| similar_policy_disambiguation | 8 | 0.750 | 0.625 | 0.875 | 0.875 | 0.750 | 0.750 |
| multi_section | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Section MRR@5 on paraphrases rises steadily with the dense weight: 0.494, 0.688, 0.698, 0.792 (dense alone: 0.917).

### Improvements and regressions per question

A question counts as improved or regressed when its Section Hit@1, Recall@5 or MRR@5 changes.

| Configuration | vs Dense | vs RRF .5/.5 |
|---|---|---|
| RRF .5/.5 | improved Q015 Q016 Q020 Q021 Q036 Q049; regressed Q008 Q010 Q012 Q013 Q041 Q047 | — |
| RRF .6/.4 | improved Q015 Q016 Q020 Q021 Q036 Q049; regressed Q010 Q013 Q047 | improved Q008 Q010 Q012 Q013 Q041 Q047; **no regressions** |
| RRF .7/.3 | improved Q015 Q016 Q021 Q049; regressed Q010 Q013 Q047 | improved the same 6; regressed Q020 Q036 |
| RRF .8/.2 | improved Q015 Q016 Q021 Q049; regressed Q010 Q047 | improved the same 6; regressed Q020 Q036 |

Rank of each relevant section (chunk rank in the full ranking; "-" means BM25 never returned it):

| Question | Section | Dense | BM25 | .5/.5 | .6/.4 | .7/.3 | .8/.2 |
|---|---|---|---|---|---|---|---|
| Q020 FIN-205.3 | FIN-205.3 | 62 | 1 | 5 | 5 | 9 | 22 |
| Q010 stolen laptop | SEC-305.4 | 1 | 20 | 6 | 4 | 4 | 3 |
| | SEC-304.2 | 4 | - | 21 | 21 | 15 | 11 |
| Q015 SEC-301.4 | SEC-301.4 | 2 | 1 | 1 | 1 | 1 | 1 |
| Q016 HR-103.3 | HR-103.3 | 2 | 1 | 1 | 1 | 1 | 1 |
| Q021 SEC-306.2 | SEC-306.2 | 2 | 1 | 1 | 1 | 1 | 1 |
| Q033 refund deadlines | REF-728.1 | 3 | 2 | 2 | 2 | 2 | 3 |
| | REF-729.1 | 2 | 5 | 3 | 3 | 3 | 2 |
| Q036 store credit | REF-729.3 | 2 | 2 | 1 | 1 | 2 | 2 |
| Q049 warranty postage | WAR-731.3 | 3 | 4 | 2 | 2 | 2 | 2 |
| Q051 purchasing | FIN-202.2 | 3 | 20 | 8 | 8 | 6 | 4 |
| | FIN-202.3 | 7 | 6 | 5 | 5 | 5 | 7 |
| | FIN-202.4 | 1 | 1 | 1 | 1 | 1 | 1 |

What the per-question results show:

- **Paraphrases improve as the dense weight rises, and identifiers suffer once it is high.** Q015, Q016 and Q021 stay at rank 1 under every weighting, because dense already ranks them 2nd. Q020, where dense ranks the answer 62nd, gets worse as the BM25 weight shrinks: rank 5, then 9, then 22. Q036 loses rank 1 from 0.7 / 0.3 onwards.
- **Q010 is never fixed.** SEC-304.2 has no BM25 match, so under 0.8 / 0.2 it scores 0.8/64 = 0.0125. That is still less than a chunk ranked 10th by both retrievers (1/70 = 0.0143). While BM25 has any weight, missing from its list is a penalty.
- **Q033 and Q051 are unchanged in their metrics** under every weighting. Q051's FIN-202.3 and FIN-202.2 trade places in and out of the top 5.
- **Hit@1 is not monotonic in the weight.** 0.6 / 0.4 and 0.8 / 0.2 both reach 31/37, while 0.7 / 0.3 reaches 30/37. This is a sign of how small the differences are, not of a real optimum at 0.6 or 0.8.

### Recommendation (cautious)

- **Keep dense as the default method.** The best weighted configurations beat it by one question on Hit@1 (31 vs 30 of 37). That is within noise for this set, and each of them still regresses 2–3 paraphrases that dense gets right.
- **If a hybrid is kept, 0.6 / 0.4 is better supported than 0.5 / 0.5.** It improved 6 questions over equal RRF and regressed none, it has the best Recall@5 (0.977), and it keeps every identifier gain. 0.8 / 0.2 has a slightly higher MRR (0.901 vs 0.897) but gives up Q020 and Q036.
- **No defaults were changed.** Dense remains the default method and 0.5 / 0.5 the default hybrid weighting. Any choice should be confirmed once on the held-out test split, and that run has not been done.

### Limitations of choosing weights on 37 questions

- One question is worth 0.027 of Hit@1. Every difference among the four weightings is three questions or fewer.
- Choosing the best of four configurations on the same development set makes the winner's development score optimistic. The held-out test split exists to correct for this.
- The best weighting depends on the question mix (here 7 identifier questions against 8 paraphrases) and would shift with a different corpus or set of questions.
- Weighting cannot fix the core limitation: a single global weight cannot be right for both kinds of question. Whether reranking (M3) does better is a separate experiment.

**Also fixed in M2.5:** `bm25_scores` returned NaN for chunks without the query term when `k1 = 0`, because 0/0 is undefined. Now only chunks containing the term are scored. Scores at the default k1 = 1.5 are bit-for-bit unchanged, as the BM25 regression test confirms.

## Future Milestones (not implemented)

- **M3**: Neural cross-encoder reranking
- **M4**: LLM-generated answers with source citations
- **M5**: Hallucination detection and evidence verification
- **M6**: Fine-tuning a small transformer-based evidence verifier
- **M7**: Prompt-injection attacks and defenses
- **M8**: Automated evaluation and ablation studies
