import argparse

from embeddings_workshop.utils.logger import get_logger
from embeddings_workshop.utils.time import timed_function

from .chroma_store import get_retriever, vector_store, get_hyde_retriever


@timed_function
def invoke_query(query: str = "", k: int = 3):
    log = get_logger("query_chroma")

    if query == "":
        log.error(f'Invalid query "{query}"')
        return

    retriever = get_retriever(k)

    log.info(f"Searching for a response for {query}")

    response = retriever.invoke(query)

    return response

@timed_function
def direct_query(query: str, k: int = 3):
    results = vector_store.similarity_search_with_score(query, k=k)
    return results

@timed_function
def hyde_query(query: str, k: int =3):
    retriever = get_hyde_retriever(k)
    
    response = retriever.invoke(query)
    return response

# Simple function to query chroma db
def query_db(argv: list[str] | None = None):
    log = get_logger("query_db")
    log.info("Processing query")

    # Process params

    parser = argparse.ArgumentParser(description="Ingest document")
    parser.add_argument("-q", "--query", help="file or folder to ingest")
    parser.add_argument(
        "-k", "--responses", help="Number of responses", type=int, default=3
    )
    parser.add_argument(
        "-m",
        "--method",
        help="Method to use: invoke or direct query",
        type=str,
        default="invoke",
    )
    args = parser.parse_args(argv)
    log.info(f"Using method {args.method}")
    if args.method == "invoke":
        response = invoke_query(args.query, args.responses)

        log.info("Responses:")
        for res in response:
            print("\n--------------------------\n")
            print(f"Page: {res.metadata['page_label']}")
            print(f"Content: {res.page_content}")

    elif args.method == "query":
        response = direct_query(args.query, args.responses)            

        # Repsonses are tuples
        for res in response:
            doc, distance = res
            print("\n--------------------------\n")
            print(f"Page: {doc.metadata['page_label']}")
            print(f"Content: {doc.page_content}")
            print(f"Distance: {distance}")  # The smaller the better
            
    elif args.method == "hyde":
        response = hyde_query(args.query,args.responses)
        for res in response:
            print("\n--------------------------\n")
            print(f"Page: {res.metadata['page_label']}")
            print(f"Content: {res.page_content}")
    else:
        log.error("Invalid method recieved")
