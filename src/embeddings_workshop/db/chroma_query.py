import argparse

from embeddings_workshop.utils.logger import get_logger
from .chroma_store import get_retriever

def query(query:str = '', k:int = 3):
    log = get_logger('query_chroma')
    
    if query == '':
        log.error(f'Invalid query "{query}"')
        return
    
    retriever = get_retriever(k)
    
    log.info(f'Searching for a response for {query}')
    
    response = retriever.invoke(query)
    
    return response


# Simple function to query chroma db
def query_db(argv: list[str] | None = None):
    log = get_logger("query_db")
    log.info('Processing query')
    
    # Process params
    
    parser = argparse.ArgumentParser(description="Ingest document")
    parser.add_argument("-q", "--query", help="file or folder to ingest")    
    args = parser.parse_args(argv)
    
    response = query(args.query,2)
    
    log.info('Respuesta:')
    for res in response:
        print('\n--------------------------\n')
        print(f'Page: {res.metadata['page_label']}')
        print(f'Content: {res.page_content}')        