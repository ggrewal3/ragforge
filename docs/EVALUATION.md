# RAGForge evaluation summary (v1.0)

All numbers below were measured in this repository. Each section says what data was used and what the result does **not** show. Detailed method notes and per-question analyses are in [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md).

**Data caveat that applies throughout.** The knowledge base is 24 **synthetic** company-policy documents (90 sections, 92 chunks) written for this project, and every benchmark and training example is derived from them. Results describe behaviour on this small, clean, in-house corpus. They do not establish performance on real policy documents or in production.

## 1. Retrieval (M1–M3)

55 hand-labelled questions with gold sections, split 37 development / 18 held-out. All comparisons below use the **development split only**; the held-out retrieval split was not used to report or choose between methods.

| Development (37 q) | Dense | BM25 | RRF 0.5/0.5 | **RRF 0.6/0.4** | Hybrid 0.6/0.4 → cross-encoder |
|---|---|---|---|---|---|
| Section Hit@1 | 0.811 | 0.676 | 0.757 | **0.838** | 0.838 |
| Section Recall@5 | 0.964 | 0.901 | 0.964 | **0.977** | 0.919 |
| Section MRR@5 | 0.887 | 0.782 | 0.842 | **0.897** | 0.883 |
| Query latency (mean) | 4.3 ms | 0.09 ms | 4.6 ms | 4.5 ms | 27.5 ms |

- Dense retrieval is strong on paraphrases but weak on exact section identifiers (Hit@1 0.429 on 7 identifier questions); BM25 is the reverse (1.000 on identifiers, 0.125 on paraphrases).
- Weighted RRF (dense 0.6 / BM25 0.4) is the RAG default: it had the best development scores on every section metric.
- Cross-encoder reranking tied on Hit@1 but lost relevant sections from the top 5 and cost ~23 ms more per query, so it stays optional.
- Differences are often one or two questions; categories with 3 questions move by 0.33 per question.

## 2. Grounded generation (M4)

Gemini returns structured JSON (`answer`, `citations`, `abstained`); outputs are schema-validated and every citation must name a supplied section. Offline tests and a scripted dry run exercise every failure path. A citation that passes this check only proves the section was supplied, **not** that it supports the answer — which is why claim verification exists. No systematic live evaluation of answer quality was run.

## 3. NLI verifier fine-tuning (M6.1–M6.3)

**Dataset (`v0.1-exp`).** 275 hand-authored policy facts generate 1,270 evidence–claim examples (SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE) with sentence and full-section evidence, split by policy family (884 train / 190 validation / 196 test, no family shared between splits). Labels were drafted with AI assistance; 94 were reviewed by a human, and a 50-example random audit accepted 50/50 (95% CI 0.929–1.0). It is an *experimental* freeze, so scores measure agreement with mostly AI-drafted labels.

| Model | Validation accuracy | Validation macro-F1 | Validation false-SUPPORTED |
|---|---|---|---|
| Zero-shot `nli-MiniLM2-L6-H768` (V1) | 0.626 | 0.618 | 6 (0.059) |
| Frozen encoder, trained head (V2) | 0.742 | 0.706 | 9 (0.089) |
| **Full fine-tuning (V3, selected)** | **0.874** | **0.849** | 9 (0.089) |

V3 was selected on validation before the test split was opened, then evaluated **once** on the held-out split:

| V3 held-out test (196 examples, 3 unseen families) | Value |
|---|---|
| Accuracy | **0.934** |
| Macro-F1 | **0.925** |
| False-SUPPORTED | 1 / 105 |

**Generalisation limits.** Test families are unseen, but all text comes from the same synthetic corpus and the same authoring process; 182 of 196 test labels are AI-drafted. With 3 families per split, the higher test than validation score most likely reflects which families landed where. These numbers do not transfer to the integrated pipeline, as section 4 shows.

## 4. Local verification in the pipeline (M6.4A)

V3 run through the production verifier on the existing M5 benchmark: 40 hand-written claims (15 SUPPORTED, 16 CONTRADICTED, 9 INSUFFICIENT_EVIDENCE), each with its own 1–2 evidence sections. No Gemini calls.

| Metric | Value |
|---|---|
| Accuracy | **0.800** (32/40) |
| Macro-F1 | **0.803** |
| False-SUPPORTED (wrongly accepted) | **4 / 25** |
| False-CONTRADICTED (wrongly rejected as contradicted) | 3 / 24 |
| Accuracy, documents V3 trained on | 22 / 25 (0.880) — not independent |
| Accuracy, validation-only documents | 4 / 6 |
| Accuracy, **documents never seen in training** | **6 / 9** (0.667) |
| Latency per claim | ~6 ms median (local, Apple MPS) |

For comparison, the zero-shot model (M6.1, titled premises, contradiction-first aggregation) scored 0.375 accuracy on the same benchmark. The benchmark labels are drafts awaiting independent review, and 10 benchmark claims are near-duplicates of training claims.

**Main failure modes.**

- **Entity/number confusion (false support).** V3 accepts a claim when its number or group appears somewhere in the evidence, even attached to a different rule: "P2 cases have a 1-hour target" (1 hour is the P1 target), "annual plans can be cancelled for a full refund within 30 days" (30 days is the renewal-reminder lead time; the window is 14 days). 4 of the 8 errors were false acceptances.
- **False contradiction.** V3 reads a related section about a *different* quantity as contradicting the claim: the domestic 30-day refund window was "contradicted" by "refunds are issued within 5 business days" (a processing time); a per-diem receipt exemption by the general receipt rule. Because the decision rules let one contradiction override support (`conflicting_evidence`), a single false contradiction withholds a correct answer. This happened in 2 of the 7 benchmark examples where the comparison was possible, and in the first live run.
- **Title context.** Adding document and section titles to the premise changed 12 of 190 validation predictions (8 better, 4 worse; accuracy 0.874 → 0.895), not statistically significant (p ≈ 0.39) and with no change in false-SUPPORTED. The training format (raw text) was kept.

## 5. Targeted correction (M6.4B)

One-pass, V3-guided Gemini correction with fresh extraction and fresh V3 verification is **implemented and covered by offline tests** (fake Gemini and a scripted V3 model).

**Live refund example.** In the first live end-to-end correction run on a refund question, Gemini's correction was attempted, but the run ended `revision_unsupported` and the answer was withheld, because V3 rejected claims that the source evidence supports. The same mechanism was reproduced in M6.4C (case GC1): Gemini correctly changed a seeded "14 days" to "30 days of delivery", and V3 still rejected the corrected claim as `conflicting_evidence` because it read REF-728.3 (refund *processing* within 5 business days) as a contradiction.

## 6. Controlled correction evaluation (M6.4C) — partial

12 controlled cases were planned, 2 in each of six categories. **Only 4 were completed (GC1, GC2, IE1, IE2) before the Gemini free-tier quota (20 requests/day) stopped the run; 8 cases remain unevaluated** (false contradictions, compound claims, already-correct answers and entity/number confusion).

Both conditions used identical frozen inputs (same answer, evidence snapshot, citations and initial claims); the initial V3 verdicts were identical.

| 4 completed cases | Verification only | Verification + correction |
|---|---|---|
| Answers accepted by V3 and shown | **0 / 4** | **2 / 4** |
| …provisionally judged factually correct | – | 2 / 2 (GC2, IE2) |
| Incorrect answers shown | 0 | **0** |
| Correct answers withheld | 0 | **2** (GC1, IE1) |

- **GC1:** the corrected answer was correct but withheld (V3 false contradiction, as above).
- **IE1:** V3 also flagged a correct claim through a false contradiction; Gemini's correction became an accurate abstention ("the evidence does not say whether learning days carry over"), which removed the unsupported claim but also a correct fact.
- **Cost of correction:** about 2 extra Gemini requests (~3,300 tokens, ~1,100 of them thinking) and a median of ~13 s per corrected case, versus ~0.08 s for local verification alone. No dollar cost is reported because the model used (`gemini-3.5-flash`) has no trusted price in the project's price table.

**What this does not show.**

- **Labels are provisional.** Factual labels were written by the assistant from the policy text; no human has reviewed them. V3's verdicts were not used as ground truth.
- **Errors were deliberately seeded.** All original answers were supplied by the evaluation; the errors are not organic Gemini mistakes.
- **Four cases cannot establish reliability.** This does not show that correction improves accuracy in general, and it says nothing about production reliability.
- **The untested categories include the riskiest one.** Entity/number confusion, where V3 wrongly accepts claims, was never run.

## Overall conclusions

1. Hybrid retrieval with weighted RRF was the best retriever on the development questions.
2. Fine-tuning made a small NLI model far better on the project's policy data than zero-shot (validation macro-F1 0.618 → 0.849; held-out test 0.925), but the gain shrinks on hand-written claims and unseen documents (0.800 overall, 6/9 unseen).
3. Local verification is cheap (milliseconds, no API cost) and errs toward withholding, but it still wrongly accepted 4 of 25 non-supported benchmark claims.
4. Targeted correction can repair seeded errors, but the dominant remaining failure is V3's false contradictions, which withhold correct answers. Whether correction is worth its latency and API cost is **not established** by the four completed cases.
