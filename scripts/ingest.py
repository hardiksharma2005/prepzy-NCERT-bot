"""Build the textbook FAISS index and domain vocabulary from the downloaded PDFs.

    python scripts/download_textbook.py
    python scripts/ingest.py
"""
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf._configuration import overwrite_configuration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import config  # noqa: E402
from backend.app.cache.text import analyse  # noqa: E402
from backend.app.embeddings import LocalEmbeddings  # noqa: E402

logging.getLogger("pypdf").setLevel(logging.ERROR)
# The Heredity chapter has one page with a ~100 MB vector drawing; the NCERT PDFs are trusted.
overwrite_configuration(zlib_maximum_output_length=400_000_000)


def page_text(reader: PdfReader, code: str) -> str:
    texts = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            texts.append(page.extract_text() or "")
        except Exception as exc:  # one broken page should not lose the chapter
            print(f"   ! {code} page {i} skipped: {exc}")
    return "\n".join(texts)


def clean(text: str, title: str) -> str:
    text = text.replace("�", "-")
    text = re.sub(r"/[a-z]+\d*", " ", text)                 # glyph names like /square6
    text = re.sub(r"(.{4,60}?)\1{2,}", r"\1", text)           # "Activity 9.3Activity 9.3..."
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"(Science\s*\d{1,3}|\d{1,3})", line):
            continue                                           # running header / page number
        line = re.sub(r"^Science\s?\d{1,3}\s*", "", line)
        if line.startswith(title[:15]) and re.search(r"\s\d{1,3}$", line) and len(line) < 70:
            continue                                           # "Light - Reflection ... 137"
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)              # de-hyphenate line breaks
    text = re.sub(r"(?<![.:?!])\n(?!\n)", " ", text)           # join wrapped lines
    return re.sub(r"[ \t]+", " ", text).strip()


def main() -> None:
    chapters = json.loads(config.CHAPTERS_PATH.read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    docs: list[Document] = []
    term_df: Counter[str] = Counter()
    for number, (code, title) in enumerate(chapters.items(), start=1):
        pdf = config.DATA_DIR / "pdfs" / f"{code}.pdf"
        if not pdf.exists():
            sys.exit(f"Missing {pdf}; run scripts/download_textbook.py first")
        reader = PdfReader(pdf)
        title_ascii = title.replace("–", "-")
        text = clean("\n".join(page.extract_text() or "" for page in reader.pages), title_ascii)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(
                page_content=f"[Chapter {number}: {title}]\n{chunk}",
                metadata={"chapter": title, "chapter_no": number}))
            term_df.update(analyse(chunk).terms)
        print(f"{number:>2}. {title}: {len(reader.pages)} pages, {len(chunks)} chunks")

    print(f"Embedding {len(docs)} chunks...")
    store = FAISS.from_documents(docs, LocalEmbeddings(),
                                 distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    store.save_local(str(config.INDEX_DIR))
    vocab = {t: n for t, n in term_df.most_common() if len(t) > 2 and not t.isdigit()}
    config.VOCAB_PATH.write_text(json.dumps(vocab, indent=0), encoding="utf-8")
    print(f"Saved index to {config.INDEX_DIR} and {len(vocab)} vocabulary terms")


if __name__ == "__main__":
    main()
