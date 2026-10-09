from typing import Optional

from pydantic import Field

from mem0.configs.rerankers.base import BaseRerankerConfig


class TypeSafeRerankerConfig(BaseRerankerConfig):
    """
    Configuration for the TypeSafe (Jev) reranker.

    Attributes:
        model (str): TypeSafe model to use. Defaults to "jev-latest"; pin a version for reproducible scores.
        api_key (str): TypeSafe API key. If not provided, will try to read from TYPESAFE_API_KEY environment variable.
        base_url (str): API root. If not provided, the SDK reads TYPESAFE_BASE_URL or uses https://api.typesafe.ai.
        top_k (int): Number of top documents to return after reranking.
        batch_size (int): Maximum number of memories scored in a single request.
        max_memory_chars (int): Memories longer than this are truncated before scoring.
        timeout (float): Timeout in seconds for each request. If not provided, the SDK default is used.
    """

    model: str = Field(default="jev-latest", description="TypeSafe model to use for reranking")
    api_key: Optional[str] = Field(default=None, description="TypeSafe API key")
    base_url: Optional[str] = Field(default=None, description="TypeSafe API root")
    top_k: Optional[int] = Field(default=None, description="Number of top documents to return after reranking")
    batch_size: int = Field(default=100, gt=0, description="Maximum number of memories scored in a single request")
    max_memory_chars: int = Field(
        default=2000, gt=0, description="Memories longer than this are truncated before scoring"
    )
    timeout: Optional[float] = Field(default=None, gt=0, description="Timeout in seconds for each request")
