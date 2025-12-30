import { useQuery } from '@tanstack/react-query';
import { X, BookMarked, Loader2 } from 'lucide-react';
import { chatApi } from '../lib/api';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const { data: booksData, isLoading } = useQuery({
    queryKey: ['books'],
    queryFn: chatApi.getBooks,
  });

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`
          h-full bg-gradient-to-b from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 border-r border-gray-200 dark:border-gray-700 shadow-2xl
          transition-all duration-300 ease-in-out overflow-hidden
          ${isOpen ? 'w-80' : 'w-0'}
        `}
      >
        <div className="h-full flex flex-col min-w-[20rem]">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between bg-gradient-to-r from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20">
            <h2 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <BookMarked className="w-5 h-5 text-white" />
              </div>
              Available Books
            </h2>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
              </div>
            ) : booksData?.books ? (
              <div className="space-y-3">
                {booksData.books.map((book) => (
                  <div
                    key={book.title}
                    className="group p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl hover:border-primary-500 dark:hover:border-primary-500 hover:shadow-md transition-all duration-200 cursor-pointer"
                  >
                    <h3 className="font-semibold text-sm text-gray-900 dark:text-white mb-2 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                      {book.title}
                    </h3>
                    <div className="flex items-center gap-2">
                      <div className="px-2 py-1 bg-primary-100 dark:bg-primary-900/30 rounded-md">
                        <p className="text-xs font-medium text-primary-700 dark:text-primary-400">
                          {book.chunks.toLocaleString()} chunks
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-8">
                No books available
              </p>
            )}
          </div>

          <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900">
            <div className="text-xs text-gray-600 dark:text-gray-400">
              <p className="font-semibold mb-2 text-sm">📚 Total Books: {booksData?.total_books || 0}</p>
              <p className="text-xs">⚡ Powered by RAG + Groq LLM</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
