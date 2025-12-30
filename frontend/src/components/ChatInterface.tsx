import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Send, Loader2, BookOpen, Plus } from 'lucide-react';
import { chatApi } from '../lib/api';
import type { ChatResponse } from '../types';
import ChatMessage from './ChatMessage';

export default function ChatInterface() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<ChatResponse[]>([]);

  const mutation = useMutation({
    mutationFn: chatApi.sendMessage,
    onSuccess: (data) => {
      setMessages((prev) => [...prev, data]);
      setQuery('');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || mutation.isPending) return;

    mutation.mutate({
      query: query.trim(),
      top_k: 5,
      include_images: true,
    });
  };

  const handleNewChat = () => {
    setMessages([]);
    setQuery('');
    mutation.reset();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !mutation.isPending) {
        handleSubmit(e as any);
      }
    }
  };

  const exampleQuestions = [
    'What is transfer learning?',
    'Explain the transformer architecture',
    'How do I train a neural network with PyTorch?',
    'What are the best practices for fine-tuning LLMs?',
  ];

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-gray-50 via-white to-gray-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin">
        {messages.length === 0 ? (
          <div className="max-w-3xl mx-auto mt-12">
            <div className="text-center mb-12">
              <div className="inline-block p-4 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl shadow-lg mb-6">
                <BookOpen className="w-12 h-12 text-white" />
              </div>
              <h2 className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-primary-800 dark:from-primary-400 dark:to-primary-600 bg-clip-text text-transparent mb-4 leading-tight pb-1">
                Ask me anything about AI & ML
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400">
                I'll answer using content from O'Reilly AI books with citations
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {exampleQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(question)}
                  className="group p-5 text-left bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-700 rounded-xl hover:border-primary-500 dark:hover:border-primary-500 hover:shadow-lg hover:scale-[1.02] transition-all duration-200"
                >
                  <p className="text-sm text-gray-700 dark:text-gray-300 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                    {question}
                  </p>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((message, idx) => (
              <ChatMessage key={idx} message={message} />
            ))}
          </div>
        )}

        {mutation.isPending && (
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center gap-3 p-5 bg-gradient-to-r from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 rounded-xl border border-primary-200 dark:border-primary-800 shadow-sm">
              <Loader2 className="w-5 h-5 animate-spin text-primary-600 dark:text-primary-400" />
              <span className="text-sm font-medium text-primary-700 dark:text-primary-300">
                Searching books and generating answer...
              </span>
            </div>
          </div>
        )}

        {mutation.isError && (
          <div className="max-w-4xl mx-auto">
            <div className="p-5 bg-gradient-to-r from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-xl border border-red-200 dark:border-red-800">
              <p className="text-sm font-medium text-red-700 dark:text-red-300">
                Error: {mutation.error instanceof Error ? mutation.error.message : 'Failed to get response'}
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="border-t border-gray-200 dark:border-gray-700 p-6 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            {messages.length > 0 && (
              <button
                type="button"
                onClick={handleNewChat}
                disabled={mutation.isPending}
                className="px-4 py-4 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 disabled:bg-gray-300 dark:disabled:bg-gray-800 text-gray-700 dark:text-gray-300 font-medium rounded-xl transition-all duration-200 flex items-center gap-2 disabled:cursor-not-allowed disabled:opacity-50"
                title="Start a new conversation"
              >
                <Plus className="w-5 h-5" />
                <span className="hidden sm:inline">New Chat</span>
              </button>
            )}
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about AI, ML, or Deep Learning..."
              className="flex-1 px-6 py-4 bg-gray-50 dark:bg-gray-900 border-2 border-gray-200 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
              disabled={mutation.isPending}
            />
            <button
              type="submit"
              disabled={!query.trim() || mutation.isPending}
              className="px-6 py-4 bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-700 hover:to-primary-800 disabled:from-gray-400 disabled:to-gray-500 text-white font-medium rounded-xl shadow-lg hover:shadow-xl disabled:shadow-none transition-all duration-200 flex items-center gap-2 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
              <span className="hidden sm:inline">Send</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
