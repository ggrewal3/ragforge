"""BM25 lexical retrieval, implemented from scratch.

BM25 scores a chunk by how many query words it contains, adjusted for how
rare those words are and how long the chunk is:

    BM25(q, d) = sum over unique query terms t of
                 IDF(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

  f(t, d)   term frequency: how many times term t occurs in chunk d.
  df(t)     document frequency: how many chunks contain t at least once.
  IDF(t)    inverse document frequency: rare terms are informative, common
            ones are not. We use the Lucene / Okapi variant
                IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
            which is always positive, so a term that appears in most chunks
            still adds a little instead of subtracting.
  |d|       chunk length in tokens; avgdl is the average length over all chunks.
  k1        term-frequency saturation. The 2nd occurrence of a word adds less
            than the 1st and the 10th adds almost nothing; the contribution
            approaches IDF * (k1 + 1). k1 = 0 ignores repetition entirely.
  b         length normalization. b = 1 fully divides out chunk length,
            b = 0 ignores it. Without it, long chunks win simply by containing
            more words.

Unlike dense retrieval, BM25 has no notion of meaning: "overseas" does not
match "international". In exchange it matches exact strings such as
"FIN-205.3" or "$2,500", which embedding models tend to blur.
"""

import math
import re
from collections import Counter

import numpy as np

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
#
# Design (deterministic, no external NLP libraries):
#   1. Lowercase everything.
#   2. Policy identifiers such as "FIN-205.3" or "REF-729" are kept whole.
#      A section ID also emits its parent document ID ("fin-205.3" ->
#      "fin-205.3", "fin-205"), so a query for "FIN-205" matches all of
#      that document's sections.
#   3. Mixed letter-digit tokens stay whole ("P1", "FIDO2", "5th").
#      Numbers are kept as tokens, with thousands separators removed so
#      "$25,000" and "25000" match. Decimals are kept ("0.65").
#   4. Other text is split into alphabetic words; a small stopword list removes
#      words like "the" and "what" that carry no retrieval signal.
#   5. A minimal plural stemmer maps "refunds" -> "refund" and
#      "policies" -> "policy". There is no full stemming, so "approval" and
#      "approve" stay different tokens.
#
# Tradeoffs: no stemming beyond plurals, no synonyms, and hyphenated words
# split ("three-way" -> "three", "way"). Identifiers must match the
# PREFIX-NNN(.N) pattern; anything else is tokenized as words and numbers.

_TOKEN_PATTERN = re.compile(
    r"(?P<identifier>[a-z]{2,5}-\d{3}(?:\.\d+)?)"     # fin-205.3, ref-729
    r"|(?P<alnum>[a-z]+\d[a-z\d]*|\d+[a-z][a-z\d]*)"  # p1, fido2, 5th
    r"|(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"           # 25,000  0.65  14
    r"|(?P<word>[a-z]+)"                               # words
)

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
how i if in into is it its me my of on or our should so than that the their
them then there these they this those to up was we were what when where which
who whom why will with would you your
""".split())


def minimal_stem(word):
    """Strip English plural endings (a port of Lucene's EnglishMinimalStemmer).

    "refunds" -> "refund", "policies" -> "policy", "boxes" -> "boxe"; words
    ending in "ss" or "us" ("business", "status") are left alone.
    """
    if len(word) < 3 or not word.endswith("s"):
        return word
    second_last = word[-2]
    if second_last in ("u", "s"):
        return word
    if second_last == "e":
        if len(word) > 3 and word[-3] == "i" and word[-4] not in ("a", "e"):
            return word[:-3] + "y"
        if word[-3] in ("i", "a", "o", "e"):
            return word
    return word[:-1]


def tokenize(text):
    """Turn text into a list of BM25 tokens (see the design notes above)."""
    tokens = []
    for match in _TOKEN_PATTERN.finditer(text.lower()):
        if match.group("identifier"):
            identifier = match.group("identifier")
            tokens.append(identifier)
            if "." in identifier:
                tokens.append(identifier.split(".")[0])
        elif match.group("alnum"):
            tokens.append(match.group("alnum"))
        elif match.group("number"):
            tokens.append(match.group("number").replace(",", ""))
        else:
            word = match.group("word")
            if word not in STOPWORDS and len(word) > 1:
                tokens.append(minimal_stem(word))
    return tokens


def chunk_index_text(chunk):
    """Text indexed for a chunk: the same fields the dense retriever embeds.

    section_id carries the document ID (via the parent-ID token), so
    document_id is not added separately.
    """
    return f"{chunk['section_id']} {chunk['title']} {chunk['section_title']} {chunk['text']}"


# ---------------------------------------------------------------------------
# Index and scoring
# ---------------------------------------------------------------------------

def inverse_document_frequency(document_frequency, n_documents):
    """IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5)). Positive for every 0 <= df <= N."""
    return math.log(1.0 + (n_documents - document_frequency + 0.5) / (document_frequency + 0.5))


def build_bm25_index(texts, k1=DEFAULT_K1, b=DEFAULT_B):
    """Precompute term frequencies, document frequencies, lengths and IDF for a list of texts."""
    if not texts:
        raise ValueError("Cannot build a BM25 index from an empty collection.")
    if k1 < 0 or not 0 <= b <= 1:
        raise ValueError(f"Expected k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}.")

    term_frequencies = [Counter(tokenize(text)) for text in texts]
    lengths = np.array([sum(tf.values()) for tf in term_frequencies], dtype=np.float64)
    if lengths.sum() == 0:
        raise ValueError("No indexable tokens found in the collection.")

    document_frequency = Counter()
    for tf in term_frequencies:
        document_frequency.update(tf.keys())  # each chunk counts once per term

    n_documents = len(texts)
    return {
        "k1": k1,
        "b": b,
        "n_documents": n_documents,
        "term_frequencies": term_frequencies,
        "lengths": lengths,
        "avgdl": float(lengths.mean()),
        "document_frequency": dict(document_frequency),
        "idf": {term: inverse_document_frequency(df, n_documents) for term, df in document_frequency.items()},
    }


def bm25_scores(index, query):
    """BM25 score of every indexed text for a query. Unknown terms contribute 0."""
    k1, b, avgdl = index["k1"], index["b"], index["avgdl"]
    # Per-chunk length normalization, the denominator term that does not depend on the query.
    length_norm = k1 * (1.0 - b + b * index["lengths"] / avgdl)

    scores = np.zeros(index["n_documents"], dtype=np.float64)
    for term in dict.fromkeys(tokenize(query)):  # unique terms, query order
        idf = index["idf"].get(term)
        if idf is None:
            continue
        tf = np.array([counts.get(term, 0) for counts in index["term_frequencies"]], dtype=np.float64)
        # Only chunks containing the term score; this also avoids 0/0 when k1 = 0.
        present = tf > 0
        scores[present] += idf * tf[present] * (k1 + 1.0) / (tf[present] + length_norm[present])
    return scores


def build_chunk_bm25_index(chunks, k1=DEFAULT_K1, b=DEFAULT_B):
    """BM25 index over the retrieval chunks (same chunk universe as dense retrieval)."""
    return build_bm25_index([chunk_index_text(chunk) for chunk in chunks], k1, b)


def rank_bm25(question, index, chunks, top_k):
    """Rank chunks by BM25 for a question, highest first.

    Only chunks that share at least one term with the query (score > 0) are
    returned: a lexical retriever has no evidence about the others, so a query
    of only unknown words or stopwords returns an empty list. Ties are broken
    by chunk order (stable sort), so results are deterministic.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    if len(chunks) != index["n_documents"]:
        raise ValueError(f"Got {len(chunks)} chunks but the BM25 index has {index['n_documents']}.")

    scores = bm25_scores(index, question)
    if not np.all(np.isfinite(scores)):
        raise ValueError("BM25 scores contain NaN or infinite values.")

    order = [i for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:top_k]
    results = []
    for rank, i in enumerate(order, start=1):
        result = dict(chunks[i])
        result.update(rank=rank, score=float(scores[i]), score_type="bm25")
        results.append(result)
    return results
