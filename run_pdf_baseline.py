
from pathlib import Path
import argparse
import time

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = (
    BASE_DIR
    / "major_handbook_pdfs"
    / "MATH_MajorHandbook.pdf"
)

VECTOR_STORE = BASE_DIR / "vector_store_math"
COLLECTION = "math_handbook"

EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM = "qwen2.5:0.5b"
OLLAMA_URL = "http://127.0.0.1:11434"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5


# ============================================================
# 1. LOAD PDF
# ============================================================

def load_pdf():
    reader = PdfReader(str(PDF_PATH))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": PDF_PATH.name,
                        "page": page_number,
                    },
                )
            )

    return documents


# ============================================================
# 2. LOAD OR CREATE VECTOR DATABASE
# ============================================================

def load_and_index(embeddings):
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE),
    )

    count = db._collection.count()

    if count > 0:
        print(
            f"[OK] Loaded existing vector store "
            f"({count} chunks)"
        )
        return db

    print("[INFO] Reading Mathematics Handbook...")

    documents = load_pdf()

    print(
        f"[OK] Extracted {len(documents)} text pages"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(
        f"[INFO] Indexing {len(chunks)} chunks..."
    )

    db.add_documents(chunks)

    print("[OK] Vector database created successfully")

    return db


# ============================================================
# 3. RETRIEVAL + ANSWER GENERATION
# ============================================================

def ask(
    question,
    db,
    llm,
    top_k,
    show_context=False,
):
    start = time.perf_counter()

    results = db.similarity_search_with_score(
        question,
        k=top_k,
    )

    if not results:
        return (
            "No relevant information found in the handbook.",
            [],
            time.perf_counter() - start,
        )

    # Optional retrieval debugging
    if show_context:
        print("\n" + "=" * 58)
        print("               RETRIEVED CONTEXT")
        print("=" * 58)

        for index, (doc, distance) in enumerate(
            results,
            start=1,
        ):
            source = doc.metadata.get(
                "source", "Unknown"
            )
            page = doc.metadata.get("page", "?")

            print(
                f"\n[CHUNK {index}] "
                f"{source} | Page {page} "
                f"| Distance: {distance:.4f}"
            )

            print("-" * 58)
            print(doc.page_content)
            print("-" * 58)

        print("\n[END OF RETRIEVED CONTEXT]\n")

    # Build context for LLM
    context_parts = []

    for doc, distance in results:
        source = doc.metadata.get(
            "source", "Unknown"
        )
        page = doc.metadata.get("page", "?")

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an academic assistant for the
Applied Mathematics Major at Fulbright University Vietnam.

RULES:
1. Answer using ONLY information from the provided context.
2. Do not invent course requirements or academic policies.
3. Read the question carefully, especially negation and
   alternative requirements.
4. If the context does not provide enough information,
   explicitly say that the answer cannot be found.
5. Keep the answer concise and accurate.
6. When the question asks about alternatives, explain
   the alternative rather than repeating the original option.
7. Distinguish major-specific requirements from
   university-wide graduation requirements.
8. Do not confuse major credits with total graduation credits.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    answer = llm.invoke(prompt).strip()

    elapsed = time.perf_counter() - start

    return answer, results, elapsed


# ============================================================
# 4. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Applied Mathematics PDF RAG Chatbot"
    )

    parser.add_argument(
        "--question",
        "-q",
        default="What is the name of the major?",
        help="Question to ask the chatbot",
    )

    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_LLM,
        help="Ollama model, e.g. qwen2.5:1.5b",
    )

    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the full retrieved chunks for debugging",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Handbook not found: {PDF_PATH}"
        )

    print("\n" + "=" * 58)
    print("       APPLIED MATHEMATICS - PDF RAG CHATBOT")
    print("=" * 58)

    print(f"Handbook  : {PDF_PATH.name}")
    print(f"LLM       : {args.model}")
    print(f"Embedding : {EMBED_MODEL}")
    print(f"Top-K     : {args.top_k}")

    print("-" * 58)

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL,
    )

    db = load_and_index(embeddings)

    llm = OllamaLLM(
        model=args.model,
        base_url=OLLAMA_URL,
        temperature=0,
    )

    print(f"\nQUESTION:\n{args.question}")

    answer, sources, elapsed = ask(
        question=args.question,
        db=db,
        llm=llm,
        top_k=args.top_k,
        show_context=args.show_context,
    )

    print(f"\nANSWER:\n{answer}")

    print("\nSOURCES:")

    for doc, distance in sources:
        print(
            f"  - {doc.metadata.get('source', 'Unknown')}"
            f" | Page {doc.metadata.get('page', '?')}"
            f" | Distance: {distance:.4f}"
        )

    print(
        f"\nResponse time: {elapsed:.2f} seconds"
    )

    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
