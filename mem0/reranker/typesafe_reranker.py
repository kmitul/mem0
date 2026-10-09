import logging
import os
from typing import Any, Dict, List

from mem0.reranker.base import BaseReranker

try:
    from typesafe_sdk import Noul, TypeSafeClient

    TYPESAFE_AVAILABLE = True
except ImportError:
    TYPESAFE_AVAILABLE = False

logger = logging.getLogger(__name__)

RELEVANCE_INSTRUCTIONS = "Is this memory relevant to the query?"
RELEVANCE_CRITERIA = {
    "true": "The memory contains information that answers the query or is directly about what the query asks.",
    "false": "The memory is about something else, or only shares words or a general topic with the query.",
}


class TypeSafeReranker(BaseReranker):
    """TypeSafe (Jev) reranker implementation.

    Each memory becomes one yes/no relevance question asked against the query, and all questions in a
    batch are answered in a single request. The returned probability of "yes" is the rerank score.
    """

    def __init__(self, config):
        """
        Initialize TypeSafe reranker.

        Args:
            config: TypeSafeRerankerConfig object with configuration parameters
        """
        if not TYPESAFE_AVAILABLE:
            raise ImportError(
                "typesafe-sdk package is required for TypeSafeReranker. Install with: pip install typesafe-sdk"
            )

        self.config = config
        self.api_key = config.api_key or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TypeSafe API key is required. Set TYPESAFE_API_KEY environment variable or pass api_key in config."
            )

        self.model = config.model or "jev-latest"

        self.client = TypeSafeClient(
            api_key=self.api_key,
            model=self.model,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using TypeSafe relevance probabilities.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of reranked documents with rerank_score
        """
        if not documents:
            return documents

        # Extract text content for reranking
        doc_texts = []
        for doc in documents:
            if "memory" in doc:
                doc_texts.append(doc["memory"])
            elif "text" in doc:
                doc_texts.append(doc["text"])
            elif "content" in doc:
                doc_texts.append(doc["content"])
            else:
                doc_texts.append(str(doc))

        try:
            scores = []
            batch_size = self.config.batch_size
            for start in range(0, len(doc_texts), batch_size):
                batch = doc_texts[start : start + batch_size]
                questions = {
                    f"memory_{i}": Noul(
                        instructions={
                            "question": RELEVANCE_INSTRUCTIONS,
                            "memory": str(text)[: self.config.max_memory_chars],
                        },
                        criteria=RELEVANCE_CRITERIA,
                    )
                    for i, text in enumerate(batch)
                }
                response = self.client.system_one(state={"query": query}, questions=questions)
                scores.extend(response.nouls[f"memory_{i}"].noul for i in range(len(batch)))

            # Create reranked results
            reranked_docs = []
            for doc, score in zip(documents, scores):
                reranked_doc = doc.copy()
                reranked_doc["rerank_score"] = score
                reranked_docs.append(reranked_doc)

            # Sort by relevance score in descending order
            reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Apply top_k limit
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            elif self.config.top_k:
                reranked_docs = reranked_docs[: self.config.top_k]

            return reranked_docs

        except Exception as e:
            # Fallback to original order if reranking fails
            logger.warning("TypeSafe reranking failed, falling back to original order: %s", e)
            fallback_docs = []
            for doc in documents:
                fallback_doc = doc.copy()
                fallback_doc["rerank_score"] = 0.0
                fallback_docs.append(fallback_doc)
            final_top_k = top_k or self.config.top_k
            return fallback_docs[:final_top_k] if final_top_k else fallback_docs
