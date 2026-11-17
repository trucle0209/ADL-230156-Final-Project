#!/usr/bin/env python3

"""
Sample code of a RAG System: ask a question, get an answer based on text files.
Usage: python run_txt.py
       python run_txt.py --question "Your question here"
"""

# IMPORT LIBRARIES
import sys
from pathlib import Path
from typing import Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# RAG SETTINGS
DATA_DIR = "data_txt" # Folder that contains .txt files.
VECTOR_STORE = DATA_DIR + "/vector_store" # Folder where Chroma saves and loads vector data.
COLLECTION = "rag_documents" # Name of the Chroma collection.
CHUNK_SIZE = 1000 # Maximum characters per text chunk.
CHUNK_OVERLAP = 200 # Number of characters shared between nearby chunks.
EMBED_MODEL = "nomic-embed-text" # Ollama embedding model used to convert text to vectors (Remember to pull the embedding model by running the command: ollama pull nomic-embed-text)
LLM_MODEL = "qwen2.5:0.5b" # Ollama model name for generating answers (Remember to pull the LLM model by running the command: ollama pull qwen2.5:0.5b).
TEMPERATURE = 0.7 # Controls how random or creative the model response is (the higher, the more creative; 0.0: deterministic).
OLLAMA_URL = "http://127.0.0.1:11434" # URL of local (127.0.0.1) Ollama server. Its default port is 11434
TOP_K = 5 # Number of top-matched chunks to retrieve.
SCORE_THRESHOLD = 0.3 # Minimum average relevance score before trusting retrieved context.
SAMPLE_QUESTION = "How many students are there at Fulbright?" # Default question if user does not pass --question.

# FINAL PROMPT SENT TO THE MODEL: Replacing {context} and {question} with the actual values.
# Without this instruction, the model might rely entirely on its pretrained knowledge.
# {context} is a variable that will be replaced by retrieved chunks from the vector database.
# 1. If the retrieved documents contain the answer, use them.
# 2. If the vector database does not contain the answer, use the model’s own knowledge.
# 3. Keep answers short.
# 4. This is a useful RAG trick. It helps distinguish: Answer from retrieved documents vs. Answer from model knowledge. Note: some small models might not understand this trick.
# Answer from retrieved documents
PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on provided documents.

CONTEXT FROM DOCUMENTS:
{context}
---
QUESTION: {question}

INSTRUCTIONS:
1. If the context contains relevant information, answer based on that.
2. If the context is insufficient, use your general knowledge.
3. Be concise - 2-4 sentences unless more detail is needed.

ANSWER:"""

# FUNCTION TO READ ALL .TXT DOCUMENT FILES FROM A FOLDER
def load_text_files(directory: str):
    # List to store Document objects.
    documents = []
    # Find all *.txt files.
    txt_files = list(Path(directory).glob("*.txt"))
    # Loop through each *.txt file.
    for txt_file in txt_files:
        # Show current file name.
        print("  Loading: " + txt_file.name)
        # Open file in read mode with UTF-8 text encoding.
        with open(txt_file, 'r', encoding='utf-8') as f:
            # Read the whole file into one string.
            content = f.read()
        # Create one Document object from file content.
        doc = Document(
            # Main text content of this document.
            page_content=content,
            # Save source file name in metadata.
            metadata={'source': txt_file.name}
        )
        # Add this document to the list.
        documents.append(doc)
    # Return all loaded documents.
    return documents

# FUNCTION TO LOAD EXISTING VECTOR STORE OR CREATE A NEW ONE (IF NOT EXISTING).
def load_and_index(embeddings) -> Chroma:
    # If vector store folder already exists, reuse it.
    if Path(VECTOR_STORE).exists():
        # Loading existing index.
        print("Loading existing vector store...")
        # Return a Chroma instance connected to saved data.
        return Chroma(
            # Use configured collection name.
            collection_name=COLLECTION,
            # Use embedding model for vector operations.
            embedding_function=embeddings,
            # Folder path where vectors are stored.
            persist_directory=VECTOR_STORE
        )
    # If no index exists, build one from text files.
    print("No vector store found. Indexing documents...")
    # Find all text files in DATA_DIR.
    txt_files = list(Path(DATA_DIR).glob("*.txt"))
    # If no text file exists, stop.
    if not txt_files:
        # Explain what is missing.
        print("No text files found in '" + DATA_DIR + "'")
        # Exit with error code 1.
        sys.exit(1)
    # Load text files into Document objects.
    documents = load_text_files(DATA_DIR)
    # Create chunk splitter and split documents.
    chunks = RecursiveCharacterTextSplitter(
        # Max characters per chunk.
        chunk_size=CHUNK_SIZE,
        # Shared characters between neighboring chunks.
        chunk_overlap=CHUNK_OVERLAP
    ).split_documents(documents)
    # Show how many chunks will be indexed.
    print("  Indexing " + str(len(chunks)) + " chunks...")
    # Build and persist a new vector store from chunks.
    return Chroma.from_documents(
        # Input chunks to index.
        documents=chunks,
        # Embedding model used to create vectors.
        embedding=embeddings,
        # Name of vector collection.
        collection_name=COLLECTION,
        # Folder where vector data is saved.
        persist_directory=VECTOR_STORE
    )

# FUNCTION TO RETRIEVE RELEVANT CHUNKS AND GENERATE AN ANSWER
def ask(question: str, db: Chroma, llm: OllamaLLM) -> Dict[str, Any]:
    # Retrieve top-K similar chunks with distance scores.
    raw_results = db.similarity_search_with_score(question, k=TOP_K)
    # Convert raw retrieval output into easier dictionaries.
    sources = [
        {
            # Source file name from metadata.
            'source': str(doc.metadata.get('source', 'Unknown')),
            # Convert distance to similarity score.
            'score': round(1 / (1 + score), 4),
            # Text content of retrieved chunk.
            'content': doc.page_content
        }
        # Build one source dict per retrieved result.
        for doc, score in raw_results
    ]
    # Compute average relevance score (0 if no sources).
    avg_score = sum(float(s['score']) for s in sources) / len(sources) if sources else 0
    # Use fallback if no sources or relevance is too low.
    used_fallback = not sources or avg_score < SCORE_THRESHOLD
    # Fallback branch: retrieval is not good enough.
    if used_fallback:
        # Tell model to use general knowledge.
        context = "No sufficiently relevant documents found. Use your general knowledge."
    # Retrieval branch: build context from matched chunks.
    else:
        # Join chunks with separators for readability.
        context = "\n\n---\n\n".join(
            # Format each chunk with source header and content.
            "[" + str(s['source']) + "]\n" + str(s['content']) for s in sources
        )
    # Fill prompt template placeholders with actual values.
    prompt = PROMPT_TEMPLATE.replace("{context}", context).replace("{question}", question)
    # Ask model and trim whitespace from response.
    answer = llm.invoke(prompt).strip()
    # Return answer, evidence sources, and fallback status.
    return {'answer': answer, 'sources': sources, 'used_fallback': used_fallback}

# MAIN FUNCTION
def main():
    # Parse command-line arguments for question input.
    import argparse
    # Create command-line parser with a short description
    parser = argparse.ArgumentParser(description="Simple RAG Q&A (Text Files)")
    # Add question argument with default sample question.
    parser.add_argument('--question', '-q', default=SAMPLE_QUESTION, help='Question to ask')
    # Parse command-line arguments.
    args = parser.parse_args()

    # Load embeddings and LLM
    print("Loading Ollama embeddings: " + EMBED_MODEL)
    # Initialize Ollama embedding model.
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    # Load existing or build new vector store.
    db = load_and_index(embeddings)
    # Show which LLM is being used.
    print("Connecting to Ollama (" + LLM_MODEL + ")...")
    # Initialize Ollama LLM client.
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)

    # Print user question.
    print("Q: " + args.question)
    # Get answer from RAG pipeline.
    result = ask(args.question, db, llm)
    # Print answer text.
    print("\nA: " + result['answer'] + "\n")
    # If fallback happened, show note.
    if result['used_fallback']:
        # Fallback note for transparency.
        print("(answered from model knowledge — no relevant documents found)")
    # If retrieval was relevant, show top sources.
    else:
        # # Show first 3 sources
        print("Sources:")
        for s in result['sources'][:3]:
            # Print source file and similarity score.
            print("   " + str(s['source']) + "  score=" + str(s['score']))

# RUN PROGRAM IF THIS FILE IS EXECUTED DIRECTLY.
if __name__ == "__main__":
    main()
