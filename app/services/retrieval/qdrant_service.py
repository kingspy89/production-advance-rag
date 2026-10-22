import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=30,
    check_compatibility=False,
)

def search_enterprise_knowledge(query: str, limit: int = 8, gemini_api_key: str = None):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    Validates query vector dimension against collection dimension.
    """
    try:
        query_vector = embed_query(query, gemini_api_key=gemini_api_key)

        # Dimension validation: prevent silent data corruption
        try:
            collection_info = client.get_collection(settings.QDRANT_COLLECTION)
            collection_dim = collection_info.config.params.vectors.size
            query_dim = len(query_vector)

            if collection_dim != query_dim:
                error_msg = (
                    f"❌ Embedding dimension mismatch: query={query_dim}D, "
                    f"collection={collection_dim}D. Re-ingest data with matching embedding model."
                )
                logfire.error(error_msg)
                raise ValueError(error_msg)

            logfire.info(f"✅ Dimension validation passed: {query_dim}D")
        except Exception as validation_error:
            logfire.error(f"Dimension validation failed: {validation_error}")
            raise

        # Using query_points - the modern standard for Qdrant
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True # JSON
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })

        logfire.info(f"Retrieved {len(results)} candidates with scores: {[r['score'] for r in results[:3]]}")
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []

