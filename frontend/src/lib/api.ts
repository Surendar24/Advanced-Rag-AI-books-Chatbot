import axios from 'axios';
import type { ChatRequest, ChatResponse, BooksResponse, HealthResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatApi = {
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/api/chat/', request);
    return response.data;
  },

  getBooks: async (): Promise<BooksResponse> => {
    const response = await api.get<BooksResponse>('/api/chat/books');
    return response.data;
  },
};

export const healthApi = {
  check: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  },
};

export default api;
