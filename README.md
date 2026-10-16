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
- M2 and later: not implemented.

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

### Repository layout

```
data/documents/                      24 policy documents, one JSON file each
data/evaluations/retrieval_questions.json   55 labeled questions (section labels, dev/test split)
src/ingest.py                        loading and validation
src/chunking.py                      section-aware chunking
src/embeddings.py                    Sentence Transformers wrapper + normalization
src/retrieve.py                      cosine similarity, ranking, retrieval CLI
src/evaluate.py                      benchmark: metrics, splits, failure analysis
tests/test_retrieval.py              unittest suite
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
   python -m src.retrieve                    # interactive; type 'exit' or 'quit' to stop
   python -m src.retrieve --question "What is the deadline for returning an overseas order?"
   python -m src.retrieve --top-k 3          # change the number of results
   ```
4. **Run the tests**
   ```bash
   python -m unittest discover -s tests -v                          # fast: no model needed
   RAGFORGE_MODEL_TESTS=1 python -m unittest discover -s tests -v   # also loads the embedding model
   ```
5. **Run the retrieval benchmark** (see [M1.5](#m15--retrieval-evaluation))
   ```bash
   python -m src.evaluate --split development              # development split (default)
   python -m src.evaluate --split development --failures   # plus per-question results and failure analysis
   python -m src.evaluate --split test                     # held-out split: final reporting only
   python -m src.evaluate --split all                      # all 55 questions: descriptive only
   ```
   `python -m src.retrieve --evaluate` is kept as an alias for the development-split command.

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

## Future Milestones (not implemented)

- **M2**: Hybrid retrieval with BM25 plus dense embeddings, fused with Reciprocal Rank Fusion
- **M3**: Neural cross-encoder reranking
- **M4**: LLM-generated answers with source citations
- **M5**: Hallucination detection and evidence verification
- **M6**: Fine-tuning a small transformer-based evidence verifier
- **M7**: Prompt-injection attacks and defenses
- **M8**: Automated evaluation and ablation studies
