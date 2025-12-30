from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


class MetadataFilter(BaseModel):
    """Metadata filtering options"""
    book_title: Optional[str] = Field(None, description="Filter by book title")
    chapter: Optional[str] = Field(None, description="Filter by chapter")
    min_page: Optional[int] = Field(None, description="Minimum page number")
    max_page: Optional[int] = Field(None, description="Maximum page number")


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., description="User's question", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of sources to retrieve", ge=1, le=20)
    include_images: Optional[bool] = Field(True, description="Whether to include relevant images")
    book_filter: Optional[str] = Field(None, description="Filter results by specific book title (deprecated, use metadata_filter)")
    retrieval_method: Optional[Literal["semantic", "lexical", "hybrid"]] = Field("hybrid", description="Retrieval method to use")
    use_reranking: Optional[bool] = Field(True, description="Whether to use cross-encoder reranking")
    metadata_filter: Optional[MetadataFilter] = Field(None, description="Advanced metadata filtering options")
    similarity_threshold: Optional[float] = Field(None, description="Minimum similarity threshold (0-1)")


class SourceMetadata(BaseModel):
    """Metadata for a source chunk"""
    book_title: str
    chapter: str
    page_number: int
    chunk_index: int
    citation: str
    char_count: int
    word_count: int
    token_count: int


class Source(BaseModel):
    """Source chunk with metadata and relevance score"""
    text: str
    metadata: SourceMetadata
    distance: float = Field(..., description="Distance score (lower is more relevant)")


class ImageMetadata(BaseModel):
    """Metadata for an image"""
    image_id: str
    book_title: str
    page_number: int
    image_index: int
    filename: str
    path: str
    format: str
    confidence_score: Optional[float] = Field(None, description="Confidence score for image relevance (0-1)")


class QueryMetrics(BaseModel):
    """Performance metrics for a query"""
    total_time: float
    llm_time: float
    retrieval_time: float
    num_sources: int
    num_images: int


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    query: str
    answer: str
    sources: List[Source]
    images: List[ImageMetadata]
    metrics: QueryMetrics
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    chroma_status: str
    total_documents: int
    embedding_model: str
    llm_model: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
