from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from ..models import ChatRequest, ChatResponse, ErrorResponse
from ..rag_engine import RAGEngine


router = APIRouter(prefix="/api/chat", tags=["chat"])

rag_engine = None


def initialize_rag_engine():
    """Initialize the RAG engine (called on startup)"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for asking questions about AI books.
    
    Args:
        request: ChatRequest containing the query and optional parameters
    
    Returns:
        ChatResponse with answer, sources, images, and metrics
    """
    try:
        if rag_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG engine not initialized"
            )
        
        result = rag_engine.query(
            query=request.query,
            top_k=request.top_k,
            include_images=request.include_images,
            book_filter=request.book_filter,
            retrieval_method=request.retrieval_method,
            use_reranking=request.use_reranking,
            metadata_filter=request.metadata_filter,
            similarity_threshold=request.similarity_threshold
        )
        
        return ChatResponse(**result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in chat endpoint: {error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/books")
async def get_books() -> Dict[str, Any]:
    """
    Get list of available books in the collection.
    
    Returns:
        Dictionary with list of book titles and counts
    """
    try:
        if rag_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG engine not initialized"
            )
        
        results = rag_engine.collection.get(
            include=["metadatas"]
        )
        
        books_count = {}
        for metadata in results['metadatas']:
            book = metadata['book_title']
            books_count[book] = books_count.get(book, 0) + 1
        
        books_list = [
            {"title": book, "chunks": count}
            for book, count in sorted(books_count.items())
        ]
        
        return {
            "total_books": len(books_list),
            "books": books_list
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching books: {str(e)}"
        )
