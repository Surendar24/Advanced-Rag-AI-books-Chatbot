import { useMemo, useState, useEffect } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from '@tanstack/react-table';
import { ArrowUpDown, Eye, EyeOff, Image as ImageIcon, X } from 'lucide-react';
import type { Source, ImageMetadata } from '../types';
import { truncateText } from '../lib/utils';

interface SourcesTableProps {
  sources: Source[];
  images: ImageMetadata[];
}

const columnHelper = createColumnHelper<Source>();

export default function SourcesTable({ sources, images }: SourcesTableProps) {
  const [selectedImage, setSelectedImage] = useState<ImageMetadata | null>(null);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && selectedImage) {
        setSelectedImage(null);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [selectedImage]);

  const columns = useMemo(
    () => [
      columnHelper.accessor('metadata.book_title', {
        header: 'Book',
        cell: (info) => (
          <div className="font-medium text-sm">
            {info.getValue()}
          </div>
        ),
      }),
      columnHelper.accessor('metadata.chapter', {
        header: 'Chapter',
        cell: (info) => (
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {truncateText(info.getValue(), 40)}
          </div>
        ),
      }),
      columnHelper.accessor('metadata.page_number', {
        header: 'Page',
        cell: (info) => (
          <div className="text-sm font-mono">
            {info.getValue()}
          </div>
        ),
      }),
      columnHelper.accessor('distance', {
        header: ({ column }) => (
          <button
            onClick={() => column.toggleSorting()}
            className="flex items-center gap-1 hover:text-primary-600"
          >
            Relevance
            <ArrowUpDown className="w-3 h-3" />
          </button>
        ),
        cell: (info) => {
          const score = (1 - info.getValue()) * 100;
          return (
            <div className="flex items-center gap-2">
              <div className="w-16 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-600"
                  style={{ width: `${score}%` }}
                />
              </div>
              <span className="text-xs text-gray-500">
                {score.toFixed(0)}%
              </span>
            </div>
          );
        },
      }),
      columnHelper.display({
        id: 'actions',
        header: 'Text',
        cell: ({ row }) => {
          const isExpanded = expandedRows.has(row.index);
          return (
            <button
              onClick={() => {
                const newExpanded = new Set(expandedRows);
                if (isExpanded) {
                  newExpanded.delete(row.index);
                } else {
                  newExpanded.add(row.index);
                }
                setExpandedRows(newExpanded);
              }}
              className="flex items-center gap-1 text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700"
            >
              {isExpanded ? (
                <>
                  <EyeOff className="w-4 h-4" />
                  Hide
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4" />
                  View
                </>
              )}
            </button>
          );
        },
      }),
    ],
    [expandedRows]
  );

  const table = useReactTable({
    data: sources,
    columns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto border border-gray-200 dark:border-gray-700 rounded-lg">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="px-4 py-3 text-left text-xs font-medium text-gray-700 dark:text-gray-300 uppercase tracking-wider"
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {table.getRowModel().rows.map((row) => (
              <>
                <tr
                  key={row.id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-800/50"
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      className="px-4 py-3 text-gray-900 dark:text-gray-100"
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
                {expandedRows.has(row.index) && (
                  <tr>
                    <td
                      colSpan={columns.length}
                      className="px-4 py-3 bg-gray-50 dark:bg-gray-800/50"
                    >
                      <div className="text-sm text-gray-700 dark:text-gray-300">
                        <p className="font-medium mb-2">Source Text:</p>
                        <p className="whitespace-pre-wrap bg-white dark:bg-gray-900 p-3 rounded border border-gray-200 dark:border-gray-700">
                          {row.original.text}
                        </p>
                        <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                          Citation: {row.original.metadata.citation}
                        </p>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>

      {images.length > 0 && (
        <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <ImageIcon className="w-4 h-4" />
            Relevant Images ({images.length})
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {images.map((image) => (
              <div
                key={image.image_id}
                className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-800 relative"
              >
                {image.confidence_score !== undefined && (
                  <div className="absolute top-2 right-2 z-10">
                    <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                      image.confidence_score >= 0.7 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : image.confidence_score >= 0.5
                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                        : 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
                    }`}>
                      {(image.confidence_score * 100).toFixed(0)}% match
                    </span>
                  </div>
                )}
                <div 
                  className="aspect-video bg-gray-100 dark:bg-gray-900 flex items-center justify-center cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors"
                  onClick={() => setSelectedImage(image)}
                  title="Click to view"
                >
                  <img
                    src={`http://localhost:8000/images/${image.filename}`}
                    alt={`${image.book_title} - Page ${image.page_number}`}
                    className="max-w-full max-h-full object-contain"
                    loading="lazy"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.style.display = 'none';
                      const parent = target.parentElement;
                      if (parent && !parent.querySelector('.error-message')) {
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'error-message text-center p-4';
                        errorDiv.innerHTML = `
                          <div class="text-gray-400 dark:text-gray-500 text-sm">
                            <svg class="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            <p>Image not found</p>
                          </div>
                        `;
                        parent.appendChild(errorDiv);
                      }
                    }}
                  />
                </div>
                <div className="p-3">
                  <p className="text-xs font-medium text-gray-900 dark:text-white mb-1 truncate">
                    {image.book_title}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Page {image.page_number}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/95 p-8"
          onClick={() => setSelectedImage(null)}
        >
          <button
            onClick={() => setSelectedImage(null)}
            className="absolute top-4 right-4 p-2 text-white hover:text-gray-300 transition-colors bg-black/50 rounded-full"
            title="Close (Esc)"
          >
            <X className="w-6 h-6" />
          </button>
          
          <div 
            className="relative bg-white dark:bg-gray-900 rounded-lg shadow-2xl max-w-full max-h-full overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 sticky top-0 z-10">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white">
                {selectedImage.book_title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Page {selectedImage.page_number}
                {selectedImage.confidence_score && (
                  <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-semibold bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200">
                    {(selectedImage.confidence_score * 100).toFixed(0)}% match
                  </span>
                )}
              </p>
            </div>
            
            <div className="p-4 bg-gray-50 dark:bg-gray-800">
              <img
                src={`http://localhost:8000/images/${selectedImage.filename}`}
                alt={`${selectedImage.book_title} - Page ${selectedImage.page_number}`}
                className="w-auto h-auto"
                style={{ maxWidth: '100%', maxHeight: 'calc(100vh - 200px)' }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
