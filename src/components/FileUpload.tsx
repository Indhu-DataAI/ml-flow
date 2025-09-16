import React, { useCallback, useState } from 'react';
import { Upload, File, X } from 'lucide-react';
import apiService from '../services/api';

interface FileUploadProps {
  onFileSelect: (file: File, data: any[], columns: any[]) => void;
  selectedFile: File | null;
  onRemoveFile: () => void;
}

export default function FileUpload({ onFileSelect, selectedFile, onRemoveFile }: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && files[0].name.endsWith('.csv')) {
      handleFileUpload(files[0]);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setError(null);
    
    try {
      const response = await apiService.uploadDataset(file);
      onFileSelect(file, response.data, response.columns);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload file');
    } finally {
      setIsUploading(false);
    }
  };

  if (selectedFile) {
    return (
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <File className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900">{selectedFile.name}</h3>
              
            </div>
          </div>
          <button
            onClick={onRemoveFile}
            className="w-8 h-8 bg-red-100 hover:bg-red-200 rounded-lg flex items-center justify-center transition-colors"
          >
            <X className="w-4 h-4 text-red-600" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${
        isUploading
          ? 'border-blue-400 bg-blue-50 opacity-75'
          : isDragOver
          ? 'border-blue-400 bg-blue-50'
          : 'border-gray-300 hover:border-gray-400'
      }`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <Upload className="w-8 h-8 text-blue-600" />
      </div>
      <h3 className="text-lg font-semibold text-gray-900 mb-2">
        {isUploading ? 'Uploading...' : 'Upload your dataset'}
      </h3>
      <p className="text-gray-600 mb-6">
        Drag and drop your CSV file here, or click to browse
      </p>
      
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}
      
      <input
        type="file"
        accept=".csv"
        onChange={handleFileInput}
        disabled={isUploading}
        className="hidden"
        id="file-input"
      />
      <label
        htmlFor="file-input"
        className={`inline-flex items-center px-6 py-3 rounded-lg transition-colors cursor-pointer ${
          isUploading
            ? 'bg-blue-400 text-white cursor-not-allowed'
            : 'bg-blue-600 text-white hover:bg-blue-700'
        }`}
      >
        Choose File
      </label>
    </div>
  );
}