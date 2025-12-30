from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from pathlib import Path
from dotenv import load_dotenv

from .models import HealthResponse, ErrorResponse
from .routes import chat


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    print("\n" + "="*80)
    print("Starting AI Book Chatbot Backend")
    print("="*80)
    
    chat.initialize_rag_engine()
    
    print("="*80)
    print("Backend ready to accept requests")
    print("="*80 + "\n")
    
    yield
    
    print("\nShutting down AI Book Chatbot Backend...")


app = FastAPI(
    title="AI Book Chatbot API",
    description="RAG-based chatbot for answering questions about AI books with citations and images",
    version="1.0.0",
    lifespan=lifespan
)

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = Path(os.getenv("BASE_DIR", "d:/AI Book RAG"))
images_dir = base_dir / "data" / "images"
if images_dir.exists():
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

app.include_router(chat.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Book Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    try:
        if chat.rag_engine is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=ErrorResponse(
                    error="Service Unavailable",
                    detail="RAG engine not initialized"
                ).dict()
            )
        
        health_status = chat.rag_engine.get_health_status()
        return HealthResponse(**health_status)
    
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                detail=str(e)
            ).dict()
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc)
        ).dict()
    )
