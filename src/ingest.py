"""Load and validate the policy documents stored as JSON files.

Ingestion only reads and checks documents. It knows nothing about chunking or
embeddings, so it can be tested without loading a model.
"""

import json
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

REQUIRED_DOCUMENT_FIELDS = ("document_id", "title", "department", "effective_date", "sections")
REQUIRED_SECTION_FIELDS = ("section_id", "section_title", "text")


class DocumentValidationError(ValueError):
    """Raised when a document file is malformed or inconsistent."""


def _require_nonempty_string(value, field, source):
    if not isinstance(value, str) or not value.strip():
        raise DocumentValidationError(f"{source}: field '{field}' must be a non-empty string.")
    return value.strip()


def validate_document(raw, source="<document>"):
    """Check one raw document dict and return it in the standard structure."""
    if not isinstance(raw, dict):
        raise DocumentValidationError(f"{source}: expected a JSON object, got {type(raw).__name__}.")

    missing = [field for field in REQUIRED_DOCUMENT_FIELDS if field not in raw]
    if missing:
        raise DocumentValidationError(f"{source}: missing required field(s): {', '.join(missing)}.")

    document_id = _require_nonempty_string(raw["document_id"], "document_id", source)
    title = _require_nonempty_string(raw["title"], "title", source)
    department = _require_nonempty_string(raw["department"], "department", source)
    effective_date = _require_nonempty_string(raw["effective_date"], "effective_date", source)
    try:
        date.fromisoformat(effective_date)
    except ValueError:
        raise DocumentValidationError(
            f"{source}: effective_date '{effective_date}' is not an ISO date (YYYY-MM-DD)."
        ) from None

    sections = raw["sections"]
    if not isinstance(sections, list) or not sections:
        raise DocumentValidationError(f"{source}: 'sections' must be a non-empty list.")

    clean_sections = []
    seen_section_ids = set()
    for index, section in enumerate(sections):
        where = f"{source} section {index}"
        if not isinstance(section, dict):
            raise DocumentValidationError(f"{where}: expected a JSON object.")
        missing = [field for field in REQUIRED_SECTION_FIELDS if field not in section]
        if missing:
            raise DocumentValidationError(f"{where}: missing required field(s): {', '.join(missing)}.")
        section_id = _require_nonempty_string(section["section_id"], "section_id", where)
        if section_id in seen_section_ids:
            raise DocumentValidationError(f"{where}: duplicate section_id '{section_id}'.")
        seen_section_ids.add(section_id)
        clean_sections.append({
            "section_id": section_id,
            "section_title": _require_nonempty_string(section["section_title"], "section_title", where),
            "text": _require_nonempty_string(section["text"], "text", where),
        })

    return {
        "document_id": document_id,
        "title": title,
        "department": department,
        "effective_date": effective_date,
        "sections": clean_sections,
    }


def load_documents(documents_dir=DOCUMENTS_DIR):
    """Load every *.json document in a directory, validated and sorted by document_id."""
    documents_dir = Path(documents_dir)
    if not documents_dir.is_dir():
        raise DocumentValidationError(f"Documents directory not found: {documents_dir}")

    paths = sorted(documents_dir.glob("*.json"))
    if not paths:
        raise DocumentValidationError(f"No .json documents found in {documents_dir}")

    documents = []
    seen_ids = {}
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DocumentValidationError(f"{path.name}: invalid JSON ({error}).") from None

        document = validate_document(raw, source=path.name)
        if document["document_id"] in seen_ids:
            raise DocumentValidationError(
                f"Duplicate document_id '{document['document_id']}' in "
                f"{seen_ids[document['document_id']]} and {path.name}."
            )
        seen_ids[document["document_id"]] = path.name
        documents.append(document)

    return sorted(documents, key=lambda document: document["document_id"])
