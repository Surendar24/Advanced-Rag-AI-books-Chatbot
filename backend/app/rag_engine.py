import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from collections import defaultdict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from rank_bm25 import BM25Okapi
import numpy as np

from .models import Source, SourceMetadata, ImageMetadata, QueryMetrics, MetadataFilter


class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine for AI Book Chatbot.
    Handles embedding generation, vector search, and LLM-based answer generation.
    """
    
    def __init__(self):
        """Initialize the RAG engine with all necessary components"""
        load_dotenv()
        
        self.base_dir = Path(os.getenv("BASE_DIR", "d:/AI Book RAG"))
        self.chroma_dir = self.base_dir / "chroma_db"
        self.images_metadata_file = self.base_dir / "data" / "extracted" / "images_metadata.json"
        
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.reranker_model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.groq_model = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
        self.collection_name = "ai_books_collection"
        
        self.bm25_index = None
        self.corpus_documents = []
        self.corpus_ids = []
        self.corpus_metadatas = []
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model, ChromaDB, and Groq client"""
        print("Initializing RAG Engine...")
        
        print(f"  Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        print(f"  Loading reranker model: {self.reranker_model_name}")
        self.reranker = CrossEncoder(self.reranker_model_name)
        
        print(f"  Connecting to ChromaDB at: {self.chroma_dir}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_collection(name=self.collection_name)
        print(f"  Collection loaded: {self.collection.name} ({self.collection.count()} documents)")
        
        print("  Building BM25 index for lexical search...")
        self._build_bm25_index()
        print(f"  BM25 index built with {len(self.corpus_documents)} documents")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
            print(f"  Groq client initialized with model: {self.groq_model}")
        else:
            self.groq_client = None
            print("  WARNING: GROQ_API_KEY not found - LLM generation will be disabled")
        
        if self.images_metadata_file.exists():
            with open(self.images_metadata_file, 'r', encoding='utf-8') as f:
                all_images = json.load(f)
            
            self.images_metadata = []
            missing_count = 0
            for img in all_images:
                img_path = Path(img['path'])
                if img_path.exists():
                    self.images_metadata.append(img)
                else:
                    missing_count += 1
            
            print(f"  Loaded metadata for {len(self.images_metadata)} images")
            if missing_count > 0:
                print(f"  Warning: {missing_count} images referenced in metadata but not found on disk")
        else:
            self.images_metadata = []
            print("  No images metadata found")
        
        print("✓ RAG Engine initialized successfully\n")
    
    def _build_bm25_index(self):
        """Build BM25 index from ChromaDB collection for lexical search"""
        all_data = self.collection.get(include=["documents", "metadatas"])
        
        self.corpus_documents = all_data['documents']
        self.corpus_ids = all_data['ids']
        self.corpus_metadatas = all_data['metadatas']
        
        tokenized_corpus = [doc.lower().split() for doc in self.corpus_documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
    
    def _apply_metadata_filter(
        self,
        metadatas: List[Dict[str, Any]],
        metadata_filter: Optional[MetadataFilter] = None
    ) -> List[bool]:
        """
        Apply metadata filtering rules to a list of metadata dictionaries.
        
        Args:
            metadatas: List of metadata dictionaries
            metadata_filter: Filtering criteria
        
        Returns:
            List of boolean values indicating which items pass the filter
        """
        if not metadata_filter:
            return [True] * len(metadatas)
        
        results = []
        for metadata in metadatas:
            passes = True
            
            if metadata_filter.book_title and metadata.get('book_title') != metadata_filter.book_title:
                passes = False
            
            if metadata_filter.chapter and metadata.get('chapter') != metadata_filter.chapter:
                passes = False
            
            if metadata_filter.min_page and metadata.get('page_number', 0) < metadata_filter.min_page:
                passes = False
            
            if metadata_filter.max_page and metadata.get('page_number', float('inf')) > metadata_filter.max_page:
                passes = False
            
            results.append(passes)
        
        return results
    
    def _lexical_search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[MetadataFilter] = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Perform BM25 lexical search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filtering
        
        Returns:
            List of tuples (document, metadata, score)
        """
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        filter_mask = self._apply_metadata_filter(self.corpus_metadatas, metadata_filter)
        
        filtered_results = []
        for idx, (score, passes_filter) in enumerate(zip(scores, filter_mask)):
            if passes_filter and score > 0:
                filtered_results.append((
                    self.corpus_documents[idx],
                    self.corpus_metadatas[idx],
                    float(score)
                ))
        
        filtered_results.sort(key=lambda x: x[2], reverse=True)
        return filtered_results[:top_k]
    
    def _semantic_search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[MetadataFilter] = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Perform semantic vector search using embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filtering
        
        Returns:
            List of tuples (document, metadata, similarity_score)
        """
        query_embedding = self.embedding_model.encode(query).tolist()
        
        where_filter = None
        if metadata_filter:
            where_filter = {}
            if metadata_filter.book_title:
                where_filter["book_title"] = metadata_filter.book_title
            if metadata_filter.chapter:
                where_filter["chapter"] = metadata_filter.chapter
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        semantic_results = []
        for doc, metadata, distance in zip(chunks, metadatas, distances):
            similarity = 1.0 / (1.0 + distance)
            
            if metadata_filter:
                if metadata_filter.min_page and metadata.get('page_number', 0) < metadata_filter.min_page:
                    continue
                if metadata_filter.max_page and metadata.get('page_number', float('inf')) > metadata_filter.max_page:
                    continue
            
            semantic_results.append((doc, metadata, similarity))
        
        return semantic_results
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, Dict[str, Any], float]],
        lexical_results: List[Tuple[str, Dict[str, Any], float]],
        k: int = 60,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Combine semantic and lexical results using weighted Reciprocal Rank Fusion (RRF).
        
        Weighted RRF formula: 
        RRF_score = semantic_weight * (1 / (k + semantic_rank)) + lexical_weight * (1 / (k + lexical_rank))
        
        Args:
            semantic_results: Results from semantic search
            lexical_results: Results from lexical search
            k: Constant for RRF (default 60, standard value)
            semantic_weight: Weight for semantic results (default 0.7 = 70%)
            lexical_weight: Weight for lexical results (default 0.3 = 30%)
        
        Returns:
            Fused and ranked results
        """
        rrf_scores = defaultdict(float)
        doc_metadata_map = {}
        
        for rank, (doc, metadata, _) in enumerate(semantic_results, start=1):
            doc_id = f"{metadata['book_title']}_{metadata['chapter']}_{metadata['page_number']}_{metadata.get('chunk_index', 0)}"
            rrf_scores[doc_id] += semantic_weight * (1.0 / (k + rank))
            if doc_id not in doc_metadata_map:
                doc_metadata_map[doc_id] = (doc, metadata)
        
        for rank, (doc, metadata, _) in enumerate(lexical_results, start=1):
            doc_id = f"{metadata['book_title']}_{metadata['chapter']}_{metadata['page_number']}_{metadata.get('chunk_index', 0)}"
            rrf_scores[doc_id] += lexical_weight * (1.0 / (k + rank))
            if doc_id not in doc_metadata_map:
                doc_metadata_map[doc_id] = (doc, metadata)
        
        fused_results = []
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            doc, metadata = doc_metadata_map[doc_id]
            fused_results.append((doc, metadata, score))
        
        return fused_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[Tuple[str, Dict[str, Any], float]],
        top_k: int = 5
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Rerank results using a cross-encoder model.
        
        Args:
            query: Original query
            results: List of (document, metadata, score) tuples
            top_k: Number of top results to return after reranking
        
        Returns:
            Reranked results with cross-encoder scores
        """
        if not results:
            return []
        
        query_doc_pairs = [(query, doc) for doc, _, _ in results]
        
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        reranked_results = []
        for (doc, metadata, _), rerank_score in zip(results, rerank_scores):
            reranked_results.append((doc, metadata, float(rerank_score)))
        
        reranked_results.sort(key=lambda x: x[2], reverse=True)
        
        return reranked_results[:top_k]
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[MetadataFilter] = None,
        use_reranking: bool = True,
        retrieval_method: str = "hybrid"
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Perform hybrid search combining semantic and lexical retrieval with optional reranking.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            metadata_filter: Optional metadata filtering
            use_reranking: Whether to apply cross-encoder reranking
            retrieval_method: "semantic", "lexical", or "hybrid"
        
        Returns:
            Tuple of (documents, metadatas, scores)
        """
        retrieval_k = top_k * 4 if use_reranking else top_k
        
        if retrieval_method == "semantic":
            results = self._semantic_search(query, retrieval_k, metadata_filter)
        elif retrieval_method == "lexical":
            results = self._lexical_search(query, retrieval_k, metadata_filter)
        else:
            semantic_results = self._semantic_search(query, retrieval_k, metadata_filter)
            lexical_results = self._lexical_search(query, retrieval_k, metadata_filter)
            results = self._reciprocal_rank_fusion(semantic_results, lexical_results)
        
        if use_reranking and results:
            results = self._rerank_results(query, results, top_k)
        else:
            results = results[:top_k]
        
        documents = [doc for doc, _, _ in results]
        metadatas = [metadata for _, metadata, _ in results]
        scores = [score for _, _, score in results]
        
        return documents, metadatas, scores
    
    def _detect_intent(self, query: str) -> str:
        """
        Detect if query is a greeting or technical question
        """
        import re
        query_lower = query.lower().strip()
        
        greeting_patterns = [
            r'\bhi\b', r'\bhello\b', r'\bhey\b', r'\bgreetings\b', 
            r'\bgood morning\b', r'\bgood afternoon\b', r'\bgood evening\b',
            r'\bhow are you\b', r'\bwhats up\b', r'\bwhat\'s up\b', r'\bsup\b',
            r'\bhow do you do\b', r'\bnice to meet you\b', 
            r'\bthanks\b', r'\bthank you\b', r'\bbye\b', r'\bgoodbye\b', 
            r'\bsee you\b'
        ]
        
        technical_keywords = [
            'what', 'how', 'why', 'when', 'where', 'explain', 'describe',
            'define', 'show', 'tell', 'difference', 'compare', 'example',
            'work', 'use', 'implement', 'train', 'model', 'algorithm'
        ]
        
        technical_terms = [
            'learning', 'neural', 'network', 'model', 'training', 'data',
            'algorithm', 'transformer', 'attention', 'embedding', 'layer',
            'gradient', 'optimization', 'classification', 'regression',
            'clustering', 'supervised', 'unsupervised', 'reinforcement',
            'deep learning', 'machine learning', 'ai', 'artificial intelligence',
            'cnn', 'rnn', 'lstm', 'gpt', 'bert', 'llm', 'rag', 'vector'
        ]
            
        has_technical_keyword = any(keyword in query_lower for keyword in technical_keywords)
        has_technical_term = any(term in query_lower for term in technical_terms)
        has_question_mark = '?' in query
        
        if has_technical_keyword or has_technical_term:
            return 'technical'
        
        if has_question_mark and len(query_lower.split()) > 1:
            return 'technical'
        
        if len(query_lower.split()) <= 3:
            for pattern in greeting_patterns:
                if re.search(pattern, query_lower):
                    return 'greeting'
        
        return 'technical'
    
    def _create_greeting_response(self) -> str:
        """Create a friendly greeting response without RAG"""
        return """Hello! I'm your AI assistant specialized in answering questions about O'Reilly AI and Machine Learning books. 

I have access to the following books:
- AI Engineering
- Applied Machine Learning and AI for Engineers
- Hands-On Large Language Models
- Hands-On Machine Learning with Scikit-Learn and PyTorch

Feel free to ask me anything about artificial intelligence, machine learning, deep learning, neural networks, or any topics covered in these books. I'll provide detailed answers with citations from the relevant chapters and pages!"""
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM"""
        return """You are an AI assistant specialized in Artificial Intelligence and Machine Learning topics.

Your role is to answer questions using ONLY the provided book content. Follow these rules strictly:

1. ONLY use information from the provided context to answer questions
2. Use NUMBERED citations [1], [2], [3] etc. in the text, like academic papers
3. Place ALL full references at the END of your answer in a "References:" section
4. IMPORTANT: Each unique source should have only ONE reference number. If you cite the same source multiple times in the text, use the SAME number
5. If relevant diagrams or images are mentioned in the context, reference them in your answer
6. If the context doesn't contain enough information to answer the question, say so clearly
7. Do NOT generate information outside the given content
8. Be concise but comprehensive
9. Use technical terminology appropriately

Citation format:
- In text: "Transfer learning is a technique [1]. This approach is useful in deep learning [1][2]."
- At the end, list each unique reference ONCE:

References:
[1] Hands-On Machine Learning with Scikit-Learn and PyTorch, Chapter 11, Page 342
[2] AI Engineering, Chapter 3, Page 45

CRITICAL: Do NOT list the same book/chapter/page combination multiple times with different numbers. Each unique source gets exactly one number.

If diagrams/figures are relevant, mention them like:
"See Figure 4.5 for the RAG architecture [1]."
"""
    
    def _consolidate_chunks(self, chunks: List[str], metadatas: List[Dict[str, Any]]) -> tuple:
        """Consolidate multiple chunks from the same book/chapter/page"""
        consolidated = {}
        
        for chunk, metadata in zip(chunks, metadatas):
            key = (metadata['book_title'], metadata['chapter'], metadata['page_number'])
            
            if key not in consolidated:
                consolidated[key] = {
                    'book_title': metadata['book_title'],
                    'chapter': metadata['chapter'],
                    'page_number': metadata['page_number'],
                    'chunk_index': metadata.get('chunk_index', 0),
                    'char_count': 0,
                    'word_count': 0,
                    'token_count': 0,
                    'chunks': []
                }
            
            consolidated[key]['chunks'].append(chunk)
            consolidated[key]['char_count'] += metadata.get('char_count', len(chunk))
            consolidated[key]['word_count'] += metadata.get('word_count', len(chunk.split()))
            consolidated[key]['token_count'] += metadata.get('token_count', len(chunk) // 4)
        
        consolidated_chunks = []
        consolidated_metadatas = []
        
        for key, data in consolidated.items():
            combined_text = '\n\n'.join(data['chunks'])
            consolidated_chunks.append(combined_text)
            consolidated_metadatas.append({
                'book_title': data['book_title'],
                'chapter': data['chapter'],
                'page_number': data['page_number'],
                'chunk_index': data['chunk_index'],
                'citation': f"[{data['book_title']}, {data['chapter']}, Page {data['page_number']}]",
                'char_count': data['char_count'],
                'word_count': data['word_count'],
                'token_count': data['token_count']
            })
        
        return consolidated_chunks, consolidated_metadatas
    
    def _format_context(self, chunks: List[str], metadatas: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context for the LLM"""
        consolidated_chunks, consolidated_metadatas = self._consolidate_chunks(chunks, metadatas)
        
        context_parts = []
        
        for i, (chunk, metadata) in enumerate(zip(consolidated_chunks, consolidated_metadatas), 1):
            context_part = f"""Source {i}:
Book: {metadata['book_title']}
Chapter: {metadata['chapter']}
Page: {metadata['page_number']}
Citation: {metadata['citation']}

Content:
{chunk}

---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _find_relevant_images(
        self,
        query: str,
        chunks: List[str],
        chunks_metadata: List[Dict[str, Any]],
        max_images: int = 3,
        confidence_threshold: float = 0.3
    ) -> List[ImageMetadata]:
        """
        Find images relevant to the query with confidence scoring.
        
        Args:
            query: User's query
            chunks: Retrieved text chunks
            chunks_metadata: Metadata for chunks
            max_images: Maximum number of images to return
            confidence_threshold: Minimum confidence score (0-1) for image relevance
        
        Returns:
            List of relevant images with confidence scores
        """
        query_lower = query.lower()
        visual_keywords = [
            'show', 'diagram', 'chart', 'graph', 'plot', 'image', 
            'illustration', 'visualization', 'architecture', 'pipeline',
            'visualize', 'draw', 'picture', 'look like', 'figure'
        ]
        
        has_visual_intent = any(keyword in query_lower for keyword in visual_keywords)
        
        chunks_mention_visuals = any(
            any(keyword in chunk.lower() for keyword in ['figure', 'fig.', 'diagram', 'chart', 'table', 'image'])
            for chunk in chunks
        )
        
        if not (has_visual_intent or chunks_mention_visuals):
            return []
        
        query_embedding = self.embedding_model.encode(query)
        
        chunk_pages_map = {}
        for chunk, chunk_metadata in zip(chunks, chunks_metadata):
            book = chunk_metadata['book_title']
            page = chunk_metadata['page_number']
            key = (book, page)
            
            if key not in chunk_pages_map:
                chunk_pages_map[key] = {
                    'text': chunk,
                    'embedding': self.embedding_model.encode(chunk)
                }
        
        scored_images = []
        
        for img in self.images_metadata:
            book = img['book_title']
            page = img['page_number']
            
            confidence_score = 0.0
            
            for page_offset in [0, -1, 1]:
                key = (book, page + page_offset)
                if key in chunk_pages_map:
                    chunk_data = chunk_pages_map[key]
                    
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        chunk_data['embedding'].reshape(1, -1)
                    )[0][0]
                    
                    if page_offset == 0:
                        confidence_score = max(confidence_score, similarity)
                    else:
                        confidence_score = max(confidence_score, similarity * 0.8)
                    
                    visual_mentions = sum(
                        1 for keyword in ['figure', 'fig.', 'diagram', 'chart', 'table']
                        if keyword in chunk_data['text'].lower()
                    )
                    if visual_mentions > 0:
                        confidence_score = min(1.0, confidence_score + 0.1 * visual_mentions)
            
            if confidence_score >= confidence_threshold:
                img_with_score = dict(img)
                img_with_score['confidence_score'] = float(confidence_score)
                scored_images.append(img_with_score)
        
        scored_images.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return [ImageMetadata(**img) for img in scored_images[:max_images]]
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        include_images: bool = True,
        book_filter: Optional[str] = None,
        retrieval_method: str = "hybrid",
        use_reranking: bool = True,
        metadata_filter: Optional[MetadataFilter] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline for a query with hybrid retrieval.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            include_images: Whether to find relevant images
            book_filter: Optional filter by book title (deprecated, use metadata_filter)
            retrieval_method: "semantic", "lexical", or "hybrid"
            use_reranking: Whether to use cross-encoder reranking
            metadata_filter: Advanced metadata filtering options
            similarity_threshold: Minimum similarity threshold for results
        
        Returns:
            Dictionary containing answer, sources, images, and metrics
        """
        start_time = time.time()
        
        intent = self._detect_intent(query)
        
        if intent == 'greeting':
            answer = self._create_greeting_response()
            total_time = time.time() - start_time
            
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "images": [],
                "metrics": QueryMetrics(
                    total_time=total_time,
                    llm_time=0,
                    retrieval_time=0,
                    num_sources=0,
                    num_images=0
                )
            }
        
        if book_filter and not metadata_filter:
            metadata_filter = MetadataFilter(book_title=book_filter)
        elif book_filter and metadata_filter and not metadata_filter.book_title:
            metadata_filter.book_title = book_filter
        
        chunks, metadatas, scores = self._hybrid_search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            use_reranking=use_reranking,
            retrieval_method=retrieval_method
        )
        
        retrieval_time = time.time() - start_time
        
        if similarity_threshold is not None:
            filtered_results = []
            for chunk, metadata, score in zip(chunks, metadatas, scores):
                if score >= similarity_threshold:
                    filtered_results.append((chunk, metadata, score))
            
            if filtered_results:
                chunks, metadatas, scores = zip(*filtered_results)
                chunks = list(chunks)
                metadatas = list(metadatas)
                scores = list(scores)
            else:
                chunks, metadatas, scores = [], [], []
        
        if not chunks:
            answer = """I apologize, but I couldn't find relevant information about your question in the available O'Reilly AI and Machine Learning books.

The books I have access to cover topics like:
- Machine Learning fundamentals and algorithms
- Deep Learning and Neural Networks
- Large Language Models (LLMs)
- AI Engineering practices
- Natural Language Processing
- Computer Vision
- Model training and deployment

Could you please rephrase your question or ask about a different AI/ML topic? I'll be happy to help with questions related to these areas!"""
            
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "images": [],
                "metrics": QueryMetrics(
                    total_time=time.time() - start_time,
                    llm_time=0,
                    retrieval_time=retrieval_time,
                    num_sources=0,
                    num_images=0
                )
            }
        
        relevant_images = []
        if include_images and self.images_metadata:
            relevant_images = self._find_relevant_images(query, chunks, metadatas)
        
        context = self._format_context(chunks, metadatas)
        consolidated_chunks, consolidated_metadatas = self._consolidate_chunks(chunks, metadatas)
        
        llm_start = time.time()
        
        if not self.groq_client:
            answer = "Error: Groq client not initialized. Please configure GROQ_API_KEY."
            llm_time = 0
        else:
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": self._create_system_prompt()},
                        {"role": "user", "content": f"""Context from AI books:

{context}

Question: {query}

Please provide a comprehensive answer based on the context above, including all relevant citations."""}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                answer = response.choices[0].message.content
                llm_time = time.time() - llm_start
                
            except Exception as e:
                answer = f"Error generating answer: {str(e)}"
                llm_time = 0
        
        total_time = time.time() - start_time
        
        sources = [
            Source(
                text=consolidated_chunk,
                metadata=SourceMetadata(**consolidated_metadata),
                distance=0.0
            )
            for consolidated_chunk, consolidated_metadata in zip(consolidated_chunks, consolidated_metadatas)
        ]
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "images": relevant_images,
            "metrics": QueryMetrics(
                total_time=total_time,
                llm_time=llm_time,
                retrieval_time=retrieval_time,
                num_sources=len(sources),
                num_images=len(relevant_images)
            )
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the RAG engine"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "chroma_status": "connected",
            "total_documents": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "llm_model": self.groq_model
        }
