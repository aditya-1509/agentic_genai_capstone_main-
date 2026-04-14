import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "rag/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def build_index(pdf_glob: str = "*.pdf", force_rebuild: bool = False) -> bool:
    """
    Build a Chroma vector store from all PDFs found in the project root.
    Returns True if built, False if skipped (already exists and not forced).
    """
    if not force_rebuild and os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print(f"Vector store already exists at {PERSIST_DIR}. Skipping rebuild.")
        return False

    pdf_files = glob.glob(pdf_glob)
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return False

    print(f"Found {len(pdf_files)} PDF(s): {[os.path.basename(p) for p in pdf_files]}")

    documents = []
    for pdf_file in pdf_files:
        print(f"  Loading {os.path.basename(pdf_file)}...")
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            # Attach clean source metadata
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf_file)
            documents.extend(docs)
        except Exception as e:
            print(f"  Warning: Could not load {pdf_file}: {e}")

    if not documents:
        print("No documents loaded from PDFs.")
        return False

    print(f"Total pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print(f"Initializing embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f"Building Chroma vector store at {PERSIST_DIR} ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print("Vector store built and persisted successfully.")
    return True


if __name__ == "__main__":
    build_index(force_rebuild=True)
