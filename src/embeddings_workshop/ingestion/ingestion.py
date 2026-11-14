import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings_workshop.db.chroma_store import save_chunks
from embeddings_workshop.embedding.embedding import get_embeddings
from embeddings_workshop.utils.logger import get_logger


def ingest_document(path: str, **kwargs):
    log = get_logger("ingestion_pipeline")
    log.info("Starting process")

    # Get arguments
    dryRun = kwargs.get("dryRun", False)

    # Load document

    file = Path(path)
    extension = file.suffix.lower()
    doc = None

    match extension:
        case ".pdf":
            # Create document out of pdf file
            log.info("Processing pdf file")
            pdfLoader = PyPDFLoader(str(file))

            # pdfData is an array of dictionaries where that hold the page_content of each page, also its metadata
            doc = pdfLoader.load()
            # print(pdfData[3])
            pass
        case _:
            # Unknown type of file
            log.error(f"Cannot process file with suffic {extension}")

    # Chunk document

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    chunks = splitter.split_documents(doc)

    log.info(f"{len(chunks)} chunks computed out of document")

    # if not dryRun:
    #     # Create embeddings
    #     content = [c.page_content for c in chunks]

    #     vectors = get_embeddings(content)

    # else:
    #     log.info("Skipping embedding creation due to dry run")

    # Save record on db

    if not dryRun:
        # Save documents

        save_chunks(chunks)
        log.info("Succesfuly loaded chunks into chroma db")
    else:
        log.info("Skipping insertion on db due to dry run")

    # End
    pass


def ingestion(argv: list[str] | None = None):
    log = get_logger("ingestion")
    log.info("Executing manual ingestion command")

    # Process params

    parser = argparse.ArgumentParser(description="Ingest document")
    parser.add_argument("-f", "--file", help="file or folder to ingest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    print("\n------------------------------------------")
    print("Arguments: ")
    print(f"Document path: {args.file}")
    print(f"Dry run: {args.dry_run}")
    print("------------------------------------------\n")

    # Validata file existance

    document = Path(args.file)
    isDryRun = args.dry_run

    if not document.exists():
        log.error(f"Document not found on path {args.file}. Cancelled execution...")

    # Document validated, execute ingestion

    ingest_document(args.file, dryRun=isDryRun)
