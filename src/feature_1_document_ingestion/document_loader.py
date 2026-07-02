# document_loader.py
# Responsibility: Read a file from disk and return its raw text content.
# Supports .txt, .md, and .pdf files.
# If you add a new file type later, just add a new elif block below.

import os
from pypdf import PdfReader


def load_document(file_path: str) -> str:
    """
    Load a document from the given file path and return its text.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        The raw text content of the file as a single string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".txt" or extension == ".md":
        return _load_txt(file_path)
    elif extension == ".pdf":
        return _load_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{extension}'. Supported: .txt, .md, .pdf"
        )


def _load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text: 
            pages.append(text)
    return "\n\n".join(pages)
