"""Turn text into embedding vectors with a Sentence Transformers model.

How text becomes a vector:
  1. Tokenization: the text is split into subword tokens ("reimbursement" ->
     "re", "##im", "##bur", ...) and mapped to integer IDs from a fixed vocabulary.
  2. Transformer encoder: the token IDs pass through self-attention layers that
     produce one contextual vector per token. Each token's vector depends on
     the tokens around it.
  3. Pooling: the token vectors are averaged (mean pooling for MiniLM) into a
     single fixed-length vector for the whole passage, called the embedding.
  4. Semantic similarity: the model was trained so that texts with similar
     meaning get vectors pointing in similar directions. The cosine of the
     angle between two embeddings is therefore a proxy for semantic similarity,
     even when the texts share no words ("overseas" vs "international").

The default model, all-MiniLM-L6-v2, truncates inputs at 256 tokens. Chunks of
about 120 words stay well under that limit, so no chunk text is silently cut.
"""

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


def load_model(model_name=DEFAULT_MODEL_NAME):
    """Load a Sentence Transformers model. Weights download to the Hugging Face cache on first use."""
    # Imported here so that modules and tests that only need NumPy logic do
    # not pay the cost of importing PyTorch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def normalize_rows(matrix):
    """Scale each row to unit L2 length. Rejects zero-norm or non-finite rows."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Embeddings contain NaN or infinite values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        bad_rows = np.flatnonzero(norms[:, 0] == 0).tolist()
        raise ValueError(f"Cannot normalize zero-norm vector(s) at row(s) {bad_rows}.")
    return matrix / norms


def chunk_embedding_text(chunk):
    """The string that is embedded for a chunk.

    The document title and section title are prepended to the chunk text so
    that a passage like "Requests must be submitted at least 30 days before
    departure" still carries the context that it belongs to the Remote Work
    Policy. This decision was made up front, not tuned against the evaluation set.
    """
    return f"{chunk['title']} - {chunk['section_title']} ({chunk['section_id']}): {chunk['text']}"


def embed_texts(model, texts, batch_size=DEFAULT_BATCH_SIZE):
    """Embed a list of strings and return an (n_texts, dim) float32 array of unit vectors."""
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("All texts to embed must be non-empty strings.")

    # encode() processes the texts in batches of `batch_size` inside the model.
    raw = model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    embeddings = np.asarray(raw, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Expected embeddings of shape ({len(texts)}, dim), got {embeddings.shape}."
        )
    # Unit length means cosine similarity reduces to a dot product (see retrieve.py).
    return normalize_rows(embeddings)


def embed_query(model, question):
    """Embed one question and return a 1-D unit vector."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    return embed_texts(model, [question.strip()])[0]


def embed_chunks(model, chunks, batch_size=DEFAULT_BATCH_SIZE):
    """Embed every chunk (title + section + text) and return an (n_chunks, dim) array."""
    return embed_texts(model, [chunk_embedding_text(chunk) for chunk in chunks], batch_size)
