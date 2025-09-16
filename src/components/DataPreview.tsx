import React from 'react';
import { Eye, EyeOff, Target, Trash2 } from 'lucide-react';

interface Column {
  name: string;
  type: string;
  included: boolean;
  isTarget: boolean;
  removed: boolean;
}

interface DataPreviewProps {
  data: any[];
  columns: Column[];
  onToggleColumn: (columnName: string) => void;
  onSetTarget: (columnName: string) => void;
  onRemoveColumn: (columnName: string) => void;
}

export default function DataPreview({ data, columns, onToggleColumn, onSetTarget, onRemoveColumn }: DataPreviewProps) {
  const activeColumns = columns.filter(col => !col.removed);
  const includedColumns = activeColumns.filter(col => col.included);
  const targetColumn = columns.find(col => col.isTarget);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Preview</h3>
        
        {/* Column Controls */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700">Column Selection</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {activeColumns.map((column) => (
              <div
                key={column.name}
                className={`flex items-center justify-between p-3 rounded-lg border-2 transition-all ${
                  column.isTarget 
                    ? 'border-green-200 bg-green-50'
                    : column.included
                    ? 'border-blue-200 bg-blue-50'
                    : 'border-gray-200 bg-gray-50'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => onToggleColumn(column.name)}
                    className={`w-5 h-5 rounded flex items-center justify-center ${
                      column.included 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-300'
                    }`}
                  >
                    {column.included ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                  </button>
                  <div>
                    <span className="text-sm font-medium text-gray-900">{column.name}</span>
                    <span className="text-xs text-gray-500 ml-1">({column.type})</span>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => onSetTarget(column.name)}
                    className={`w-6 h-6 rounded flex items-center justify-center ${
                      column.isTarget
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-200 hover:bg-gray-300'
                    }`}
                    title="Set as target variable"
                  >
                    <Target className="w-3 h-3" />
                  </button>
                  <button
                    onClick={() => onRemoveColumn(column.name)}
                    className="w-6 h-6 rounded flex items-center justify-center bg-red-100 hover:bg-red-200 text-red-600"
                    title="Remove column"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              </div>
            ))}
          </div>
          
          {targetColumn && (
            <div className="bg-green-100 border border-green-200 rounded-lg p-3">
              <p className="text-sm text-green-800">
                <strong>Target Variable:</strong> {targetColumn.name}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Removed Columns Section */}
      {columns.some(col => col.removed) && (
        <div className="p-6 border-t border-gray-200 bg-gray-50">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Removed Columns</h4>
          <div className="flex flex-wrap gap-2">
            {columns.filter(col => col.removed).map((column) => (
              <div
                key={column.name}
                className="inline-flex items-center px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm"
              >
                <span>{column.name}</span>
                <button
                  onClick={() => onRemoveColumn(column.name)}
                  className="ml-2 hover:bg-red-200 rounded-full p-0.5"
                  title="Restore column"
                >
                  <Eye className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Data Table */}
      <div className="p-6">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                {includedColumns.map((column) => (
                  <th
                    key={column.name}
                    className={`text-left py-3 px-4 font-medium ${
                      column.isTarget 
                        ? 'text-green-700 bg-green-50' 
                        : 'text-gray-700'
                    }`}
                  >
                    {column.name}
                    {column.isTarget && <span className="text-xs ml-1">(Target)</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.slice(0, 10).map((row, index) => (
                <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                  {includedColumns.map((column) => (
                    <td
                      key={column.name}
                      className={`py-3 px-4 ${
                        column.isTarget ? 'bg-green-50' : ''
                      }`}
                    >
                      {row[column.name]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-500 mt-3">
          Showing first 10 rows of {data.length} total rows
        </p>
      </div>
    </div>
  );
}