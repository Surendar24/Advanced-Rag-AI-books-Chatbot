import { useState } from 'react';
import { MessageSquare, BookOpen, Image as ImageIcon, Clock, ChevronDown, ChevronUp } from 'lucide-react';
import type { ChatResponse } from '../types';
import { formatTime } from '../lib/utils';
import SourcesTable from './SourcesTable';

interface ChatMessageProps {
  message: ChatResponse;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const [showSources, setShowSources] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center flex-shrink-0 shadow-lg">
          <MessageSquare className="w-5 h-5 text-white" />
        </div>
        <div className="flex-1 bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="font-semibold text-primary-600 dark:text-primary-400 mb-2 text-sm uppercase tracking-wide">
            Your Question
          </p>
          <p className="text-gray-800 dark:text-gray-200 text-base leading-relaxed">
            {message.query}
          </p>
        </div>
      </div>

      <div className="flex items-start gap-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center flex-shrink-0 shadow-lg">
          <BookOpen className="w-5 h-5 text-white" />
        </div>
        <div className="flex-1 bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-6 shadow-md border border-gray-200 dark:border-gray-700">
          <p className="font-semibold text-emerald-600 dark:text-emerald-400 mb-3 text-sm uppercase tracking-wide">
            AI Answer
          </p>
          <div className="prose dark:prose-invert max-w-none">
            <div className="text-gray-800 dark:text-gray-200 whitespace-pre-wrap leading-relaxed text-base">
              {message.answer}
            </div>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <Clock className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <span className="text-xs font-medium text-blue-700 dark:text-blue-300">{formatTime(message.metrics.total_time)}</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <BookOpen className="w-4 h-4 text-purple-600 dark:text-purple-400" />
              <span className="text-xs font-medium text-purple-700 dark:text-purple-300">{message.metrics.num_sources} sources</span>
            </div>
            {message.metrics.num_images > 0 && (
              <div className="flex items-center gap-2 px-3 py-1.5 bg-pink-100 dark:bg-pink-900/30 rounded-lg">
                <ImageIcon className="w-4 h-4 text-pink-600 dark:text-pink-400" />
                <span className="text-xs font-medium text-pink-700 dark:text-pink-300">{message.metrics.num_images} images</span>
              </div>
            )}
          </div>

          <button
            onClick={() => setShowSources(!showSources)}
            className="mt-4 flex items-center gap-2 px-4 py-2 text-sm font-semibold text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-lg transition-all duration-200"
          >
            {showSources ? (
              <>
                <ChevronUp className="w-4 h-4" />
                Hide Sources
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4" />
                View Sources ({message.sources.length})
              </>
            )}
          </button>

          {showSources && (
            <div className="mt-4">
              <SourcesTable sources={message.sources} images={message.images} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
