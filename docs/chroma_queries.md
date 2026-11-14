# Chroma queries


Regular flow

  1. Chroma embeds your query (or takes your query_embeddings).
  2. It searches the HNSW index for the nearest vectors. This step doesn't involves SQLite.
  3. It uses SQLite to translate the matching internal IDs into your string IDs, then fetches the document text and metadata for them.s