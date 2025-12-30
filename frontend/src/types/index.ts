export interface SourceMetadata {
  book_title: string;
  chapter: string;
  page_number: number;
  chunk_index: number;
  citation: string;
  char_count: number;
  word_count: number;
  token_count: number;
}

export interface Source {
  text: string;
  metadata: SourceMetadata;
  distance: number;
}

export interface ImageMetadata {
  image_id: string;
  book_title: string;
  page_number: number;
  image_index: number;
  filename: string;
  path: string;
  format: string;
  confidence_score?: number;
}

export interface QueryMetrics {
  total_time: number;
  llm_time: number;
  retrieval_time: number;
  num_sources: number;
  num_images: number;
}

export interface ChatRequest {
  query: string;
  top_k?: number;
  include_images?: boolean;
  book_filter?: string;
}

export interface ChatResponse {
  query: string;
  answer: string;
  sources: Source[];
  images: ImageMetadata[];
  metrics: QueryMetrics;
  timestamp: string;
}

export interface Book {
  title: string;
  chunks: number;
}

export interface BooksResponse {
  total_books: number;
  books: Book[];
}

export interface HealthResponse {
  status: string;
  version: string;
  chroma_status: string;
  total_documents: number;
  embedding_model: string;
  llm_model: string;
}
