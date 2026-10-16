# RAGForge

RAGForge is a hands-on project that builds a Retrieval-Augmented Generation (RAG) system from first principles. The long-term goal is a RAG system that is **evaluation-driven, self-correcting, and attack-resistant**:

- answers grounded in retrieved evidence, with source citations;
- detection and correction of claims the evidence does not support;
- defenses against prompt-injection content hidden in documents;
- every design choice justified by measured retrieval and answer quality.

The project deliberately avoids RAG frameworks (LangChain, LlamaIndex) and vector databases so that every step can be inspected and explained.

**Current status: Milestone 1 (M1), document ingestion and dense retrieval.** Nothing beyond M1 is implemented yet.

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
data/evaluations/retrieval_questions.json   29 labeled retrieval questions
src/ingest.py                        loading and validation
src/chunking.py                      section-aware chunking
src/embeddings.py                    Sentence Transformers wrapper + normalization
src/retrieve.py                      cosine similarity, ranking, metrics, CLI
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
5. **Run the baseline evaluation**
   ```bash
   python -m src.retrieve --evaluate
   ```

## Baseline Retrieval Evaluation

`data/evaluations/retrieval_questions.json` holds 29 hand-labeled questions in four categories: exact keyword, semantic paraphrase, exact policy identifier (such as "What does FIN-205.3 say?"), and similar-policy disambiguation (such as domestic versus international refunds).

For each question, the top 5 **chunks** are retrieved and collapsed into a ranked list of unique **document** IDs (the first occurrence keeps its rank):

- **Recall@5** is |relevant ∩ retrieved documents| / |relevant|, averaged over questions.
- **MRR@5** is 1 / (rank of the first relevant document in that list), or 0 if none appears, averaged over questions.

Results from `python -m src.retrieve --evaluate` with all-MiniLM-L6-v2, 92 chunks and a 120-word limit:

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

## Future Milestones (not implemented)

- **M2**: Hybrid retrieval with BM25 plus dense embeddings, fused with Reciprocal Rank Fusion
- **M3**: Neural cross-encoder reranking
- **M4**: LLM-generated answers with source citations
- **M5**: Hallucination detection and evidence verification
- **M6**: Fine-tuning a small transformer-based evidence verifier
- **M7**: Prompt-injection attacks and defenses
- **M8**: Automated evaluation and ablation studies
