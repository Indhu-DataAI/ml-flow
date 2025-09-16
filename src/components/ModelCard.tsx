import React from 'react';
import { Play, Download, CheckCircle, Clock, AlertCircle } from 'lucide-react';

export interface Model {
  id: string;
  name: string;
  type: string;
  status: 'idle' | 'training' | 'completed' | 'error';
  progress: number;
  metrics?: {
    accuracy?: number;
    loss: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    trainingTime: number;
    mse?: number;
    mae?: number;
    r2Score?: number;
    rmse?: number;
    crossValidation?: {
      mean: number;
      std: number;
    };
  };
}

interface ModelCardProps {
  model: Model;
  onTrain: (modelId: string) => void;
  // onDownload: (modelId: string) => void;
  onSelect: (modelId: string) => void;
  isSelected: boolean;
}

export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  onTrain,
  // onDownload,
  onSelect,
  isSelected
}) => {
  const getStatusIcon = () => {
    switch (model.status) {
      case 'training':
        return <Clock className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Play className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (model.status) {
      case 'training':
        return 'border-blue-200 bg-blue-50';
      case 'completed':
        return 'border-green-200 bg-green-50';
      case 'error':
        return 'border-red-200 bg-red-50';
      default:
        return 'border-gray-200 bg-white';
    }
  };

  return (
    <div
      className={`p-6 rounded-xl border-2 transition-all duration-200 cursor-pointer hover:shadow-lg ${
        isSelected ? 'border-blue-500 shadow-md' : getStatusColor()
      }`}
      onClick={() => onSelect(model.id)}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {getStatusIcon()}
          <div>
            <h3 className="font-semibold text-gray-900">{model.name}</h3>
            <p className="text-sm text-gray-600">{model.type}</p>
          </div>
        </div>
        
        <div className="flex gap-2">
          {model.status === 'idle' && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onTrain(model.id);
              }}
              className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              title="Train Model"
            >
              <Play className="h-4 w-4" />
            </button>
          )}
          
          {/* {model.status === 'completed' && (
            // <button
            //   onClick={(e) => {
            //     e.stopPropagation();
            //     onDownload(model.id);
            //   }}
            //   className="p-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            //   title="Download Model"
            // >
            //   <Download className="h-4 w-4" />
            // </button>
          )} */}
        </div>
      </div>

      {model.status === 'training' && (
        <div className="mb-4">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>Training Progress</span>
            <span>{model.progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${model.progress}%` }}
            />
          </div>
        </div>
      )}

      {model.metrics && model.status === 'completed' && (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Metrics</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {model.metrics.accuracy && (
              <div className="flex justify-between">
                <span className="text-gray-600">Accuracy:</span>
                <span className="font-medium">{(model.metrics.accuracy * 100).toFixed(1)}%</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-gray-600">Loss:</span>
              <span className="font-medium">{model.metrics.loss.toFixed(4)}</span>
            </div>
            {model.metrics.f1Score && (
              <div className="flex justify-between">
                <span className="text-gray-600">F1 Score:</span>
                <span className="font-medium">{(model.metrics.f1Score * 100).toFixed(1)}%</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-gray-600">Time:</span>
              <span className="font-medium">{model.metrics.trainingTime.toFixed(1)}s</span>
            </div>
          </div>
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Status:</span>
          <span className={`font-medium capitalize ${
            model.status === 'completed' ? 'text-green-600' :
            model.status === 'training' ? 'text-blue-600' :
            model.status === 'error' ? 'text-red-600' : 'text-gray-600'
          }`}>
            {model.status}
          </span>
        </div>
      </div>
    </div>
  );
};