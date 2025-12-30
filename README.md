# AI Book Chatbot with RAG Pipeline

An intelligent chatbot that answers questions about Artificial Intelligence using content from O'Reilly AI books, powered by Groq LLM and RAG (Retrieval-Augmented Generation).

## 📚 Features

- **Accurate Citations**: Every answer includes inline citations in the format [Book Title, Chapter, Page]
- **Image Support**: Displays relevant diagrams and images from the books
- **RAG Pipeline**: Uses vector embeddings for semantic search and retrieval
- **Multi-Book Reasoning**: Combines content from multiple books for comprehensive answers
- **Modern UI**: React + TypeScript frontend with TanStack Query/Table
- **Fast API Backend**: Built with FastAPI for high performance

## 📖 Books Included

1. AI Engineering
2. Applied Machine Learning and AI for Engineers
3. Hands-On Large Language Models
4. Hands-On Machine Learning with Scikit-Learn and PyTorch
5. LLM Engineers Handbook
6. NLP with Transformer Models

## 🏗️ Project Structure

```
AI Book RAG/
├── Books_pdf/                          # Source PDF files
├── notebooks/                          # Jupyter notebooks for each process
│   ├── 01_pdf_ingestion.ipynb         # Extract text, images, metadata
│   ├── 02_text_chunking.ipynb         # Split and preprocess text
│   ├── 03_embedding_vectordb.ipynb    # Generate embeddings, setup Chroma
│   └── 04_rag_pipeline_test.ipynb     # Test RAG pipeline
├── backend/                            # FastAPI backend
│   ├── app/
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── models.py                  # Pydantic models
│   │   ├── rag_engine.py              # RAG pipeline logic
│   │   └── routes/
│   │       └── chat.py                # Chat endpoints
│   └── requirements.txt
├── frontend/                           # React + TypeScript frontend
│   ├── src/
│   │   ├── components/                # React components
│   │   ├── hooks/                     # TanStack Query hooks
│   │   └── App.tsx
│   └── package.json
├── data/                               # Processed data
│   ├── extracted/                     # Extracted text and images
│   └── chunks/                        # Chunked text with metadata
├── chroma_db/                         # Vector database storage
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+
- Groq API Key (free tier available at https://console.groq.com)

### Backend Setup

1. **Clone and navigate to the project**:
   ```bash
   cd "d:\AI Book RAG"
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   copy .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

5. **Run the notebooks in order**:
   - `01_pdf_ingestion.ipynb` - Extract content from PDFs
   - `02_text_chunking.ipynb` - Chunk and preprocess text
   - `03_embedding_vectordb.ipynb` - Generate embeddings and populate vector DB
   - `04_rag_pipeline_test.ipynb` - Test the RAG pipeline

6. **Start the backend**:
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open browser**:
   ```
   http://localhost:3000
   ```

## 📝 Usage

1. Open the web interface
2. Type your AI-related question in the chat input
3. Receive answers with:
   - Inline citations [Book Title, Chapter, Page]
   - Relevant diagrams/images
   - Source snippets (optional)

### Example Questions

- "What is transfer learning and how does it work?"
- "Explain the transformer architecture"
- "What are the best practices for fine-tuning LLMs?"
- "How do I implement a neural network with PyTorch?"

## 🔧 Configuration

### RAG Parameters (in `.env`)

- `TOP_K_RESULTS`: Number of relevant chunks to retrieve (default: 5)
- `CHUNK_SIZE`: Size of text chunks in characters (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TEMPERATURE`: LLM temperature for response generation (default: 0.1)

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, fast)

Alternatives:
- `sentence-transformers/all-mpnet-base-v2` (768 dimensions, more accurate)
- `BAAI/bge-small-en-v1.5` (384 dimensions, optimized for retrieval)

## 🚢 Deployment

### Backend (Render/Railway)

1. Push code to GitHub
2. Connect repository to Render/Railway
3. Set environment variables
4. Deploy with auto-build

### Frontend (Vercel)

1. Push frontend code to GitHub
2. Import project to Vercel
3. Configure build settings:
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. Deploy

### Vector Database

- **Development**: Local Chroma DB
- **Production**: Consider Qdrant Cloud (free tier) or persist Chroma to volume

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **LLM**: Groq (Mixtral-8x7b)
- **Embeddings**: Sentence Transformers
- **Vector DB**: ChromaDB
- **PDF Processing**: PyMuPDF, PDFPlumber

### Frontend
- **Framework**: React + TypeScript
- **State Management**: TanStack Query
- **Table Display**: TanStack Table
- **Styling**: Tailwind CSS
- **Icons**: Lucide React

## 📊 Advanced Features

- ✅ Multi-book reasoning
- ✅ Inline citations with metadata
- ✅ Image/diagram extraction and display
- ✅ Source snippet viewing
- ✅ Query caching with TanStack Query
- ✅ Confidence scoring for citations
- ✅ Automatic diagram highlighting

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational purposes. Please respect the copyright of the O'Reilly books.

## 🙏 Acknowledgments

- O'Reilly Media for the excellent AI books
- Groq for providing fast LLM inference
- ChromaDB for the vector database
- TanStack for Query and Table libraries
