import React, { useState } from 'react';
import { Download, Package, FileText } from 'lucide-react';
import apiService from '../services/api';

interface ModelDownloadProps {
  isModelTrained: boolean;
  modelId?: string;
}

export default function ModelDownload({ isModelTrained, modelId }: ModelDownloadProps) {
  const [downloading, setDownloading] = useState<string | null>(null);

  const downloadFile = (data: string, filename: string) => {
  const byteCharacters = atob(data); 
  const byteNumbers = new Array(byteCharacters.length);

  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }

  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: 'application/octet-stream' });

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};



  const handleDownloadModel = async () => {
    setDownloading('model');
    try {
      console.log('Downloading model with ID:', modelId);
      const response = await apiService.downloadModel(modelId);
      console.log('Download response:', response);
      
      const url = URL.createObjectURL(response.blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = response.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download model:', error);
      alert(`Download failed: ${error|| error}`);
    } finally {
      setDownloading(null);
    }
  };

  const handleDownloadEncoders = async () => {
    setDownloading('encoders');
    try {
      const response = await apiService.downloadEncoders();
      const url = URL.createObjectURL(response.blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = response.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download encoders:', error);
    } finally {
      setDownloading(null);
    }
  };

  if (!isModelTrained) {
    return null;
  }

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
      <div className="flex items-center space-x-3 mb-4">
        <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
          <Download className="w-5 h-5 text-purple-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Download Model Assets</h3>
          <p className="text-sm text-gray-600">Save your trained model and encoders for later use</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <button
          onClick={handleDownloadModel}
          disabled={downloading === 'model'}
          className="flex items-center justify-center space-x-2 p-4 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-lg transition-colors disabled:opacity-50"
        >
          <Package className="w-5 h-5 text-blue-600" />
          <span className="font-medium text-blue-900">
            {downloading === 'model' ? 'Downloading...' : 'Download Model'}
          </span>
        </button>

        <button
          onClick={handleDownloadEncoders}
          disabled={downloading === 'encoders'}
          className="flex items-center justify-center space-x-2 p-4 bg-green-50 hover:bg-green-100 border border-green-200 rounded-lg transition-colors disabled:opacity-50"
        >
          <FileText className="w-5 h-5 text-green-600" />
          <span className="font-medium text-green-900">
            {downloading === 'encoders' ? 'Downloading...' : 'Download Encoders'}
          </span>
        </button>
      </div>
    </div>
  );
}