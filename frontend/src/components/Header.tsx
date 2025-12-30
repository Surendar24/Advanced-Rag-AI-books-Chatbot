import { Menu, BookOpen, Activity } from 'lucide-react';
import type { HealthResponse } from '../types';

interface HeaderProps {
  health?: HealthResponse;
  onMenuClick: () => void;
}

export default function Header({ health, onMenuClick }: HeaderProps) {
  return (
    <header className="h-16 bg-gradient-to-r from-white via-gray-50 to-white dark:from-gray-800 dark:via-gray-900 dark:to-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 flex items-center justify-between shadow-sm">
      <div className="flex items-center gap-4">
        <button
          onClick={onMenuClick}
          className="p-2.5 hover:bg-primary-100 dark:hover:bg-gray-700 rounded-xl transition-all duration-200 hover:scale-105"
          aria-label="Toggle menu"
        >
          <Menu className="w-6 h-6 text-gray-700 dark:text-gray-300" />
        </button>
        
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl shadow-lg">
            <BookOpen className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-primary-600 to-primary-800 dark:from-primary-400 dark:to-primary-600 bg-clip-text text-transparent">
              AI Book Chatbot
            </h1>
            <p className="text-xs text-gray-600 dark:text-gray-400 font-medium">
              Ask questions about AI & ML
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {health && (
          <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl border border-green-200 dark:border-green-800 shadow-sm">
            <Activity className="w-4 h-4 text-green-600 dark:text-green-400" />
            <span className="text-sm font-semibold text-green-700 dark:text-green-300">
              {health.total_documents.toLocaleString()} chunks
            </span>
          </div>
        )}
      </div>
    </header>
  );
}
