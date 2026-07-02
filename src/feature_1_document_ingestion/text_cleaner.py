# text_cleaner.py
# Responsibility: Clean raw document text before it is chunked.
# Garbage in = garbage chunks. This step removes noise that would
# pollute embeddings: extra blank lines, leading/trailing spaces, etc.

import re


def clean_text(raw_text: str) -> str:
    """
    Clean raw text extracted from a document.

    Steps:
        1. Collapse 3+ consecutive blank lines into 2 (preserves paragraph breaks)
        2. Strip leading/trailing whitespace from every line
        3. Strip leading/trailing whitespace from the whole document

    Args:
        raw_text: The unprocessed string from document_loader.

    Returns:
        A cleaned string ready for chunking.
    """
    # Step 1: Collapse excessive blank lines (3+ newlines → 2 newlines)
    # This keeps paragraph breaks but removes padding added by some converters
    text = re.sub(r"\n{3,}", "\n\n", raw_text)

    # Step 2: Strip trailing spaces from each line
    # PDF extractors often leave trailing spaces on every line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Step 3: Strip the whole document
    text = text.strip()

    return text
