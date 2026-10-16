"""Split documents into retrieval passages ("chunks").

Why chunk? An embedding compresses a whole passage into one vector. A long
document covering many topics produces a blurry vector, and the model also
truncates long inputs. Small, focused passages give sharper vectors and let
retrieval point at the specific rule that answers a question.

Strategy (deterministic, section-aware):
  1. Each document section becomes at least one chunk. Chunks never cross
     section or document boundaries.
  2. A section longer than `max_words` is split at sentence boundaries,
     packing whole sentences until the next one would exceed the limit.
  3. A single sentence longer than `max_words` falls back to splitting on
     whitespace, so words themselves are never cut.
"""

import re

DEFAULT_MAX_WORDS = 120

# Split after ., ! or ? when followed by whitespace. This regex is only
# used to choose split points and never removes or cuts any words.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_text(text, max_words=DEFAULT_MAX_WORDS):
    """Split text into pieces of at most `max_words` whitespace-separated words."""
    if not isinstance(max_words, int) or isinstance(max_words, bool) or max_words < 1:
        raise ValueError(f"max_words must be a positive integer, got {max_words!r}.")

    pieces = []
    current = []
    for sentence in _SENTENCE_BOUNDARY.split(text.strip()):
        words = sentence.split()
        if not words:
            continue
        if len(words) > max_words:
            # The sentence alone is too long: flush, then cut it on word boundaries.
            if current:
                pieces.append(current)
                current = []
            for start in range(0, len(words), max_words):
                pieces.append(words[start:start + max_words])
            continue
        if len(current) + len(words) > max_words:
            pieces.append(current)
            current = []
        current.extend(words)
    if current:
        pieces.append(current)

    return [" ".join(words) for words in pieces]


def chunk_document(document, max_words=DEFAULT_MAX_WORDS):
    """Turn one validated document into a list of chunk dicts."""
    chunks = []
    for section in document["sections"]:
        for part_index, text in enumerate(split_text(section["text"], max_words)):
            chunks.append({
                # e.g. "REF-729.2#0": section REF-729.2, first chunk of that section.
                "chunk_id": f"{section['section_id']}#{part_index}",
                "document_id": document["document_id"],
                "title": document["title"],
                "department": document["department"],
                "section_id": section["section_id"],
                "section_title": section["section_title"],
                "text": text,
            })
    return chunks


def chunk_documents(documents, max_words=DEFAULT_MAX_WORDS):
    """Chunk every document, preserving document order."""
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, max_words))

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs are not unique; check for duplicate section_ids across documents.")
    return chunks
