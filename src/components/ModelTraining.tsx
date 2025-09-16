import React, { useState } from 'react';
import { Play, CheckCircle, Clock, AlertCircle } from 'lucide-react';
import apiService from '../services/api';
import ModelDownload from './ModelDownload';

interface ModelTrainingProps {
  algorithm: string;
  taskType: 'classification' | 'regression';
  targetColumn: string;
  featureColumns: string[];
  onTrainComplete: (metrics: any) => void;
}

export default function ModelTraining({ algorithm, taskType, targetColumn, featureColumns, onTrainComplete }: ModelTrainingProps) {
  const [isTraining, setIsTraining] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [id,setid] = useState<string|null>(null);
  const startTraining = async () => {
    setIsTraining(true);
    setError(null);

    try {
      const result = await apiService.trainModel({
        algorithm,
        task_type: taskType,
        target_column: targetColumn,
        feature_columns: featureColumns,
      });
      
      setIsTraining(false);
      setid(result.model_id)
      
      setIsComplete(true);
      onTrainComplete(result.metrics);

    } catch (err: any) {
      setIsTraining(false);
      setError(err.response?.data?.detail || 'Training failed');
    }
  };

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Model Training</h3>
        <p className="text-gray-600">
          Ready to train your model using {algorithm.replace(/_/g, ' ').replace(/clf|reg/g, '').trim()}
        </p>
      </div>

      <div className="space-y-6">
        {!isTraining && !isComplete && (
          <div className="text-center py-8">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Play className="w-8 h-8 text-blue-600" />
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">Ready to Train</h4>
            <p className="text-gray-600 mb-6">Click the button below to start training your model</p>
            <button
              onClick={startTraining}
              className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Play className="w-4 h-4 mr-2" />
              Start Training
            </button>
          </div>
        )}

        {isTraining && (
          <div className="text-center py-8">
            <div className="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Clock className="w-8 h-8 text-yellow-600 animate-spin" />
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">Training in Progress</h4>
            <p className="text-gray-600">Training your {algorithm.replace(/_/g, ' ').replace(/clf|reg/g, '').trim()} model...</p>
          </div>
        )}

        {error && (
          <div className="text-center py-8">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <AlertCircle className="w-8 h-8 text-red-600" />
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">Training Failed</h4>
            <p className="text-gray-600 mb-6">{error}</p>
            <button
              onClick={() => {
                setError(null);
                setIsComplete(false);
              }}
              className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {isComplete && (
          <div className="text-center py-8">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="w-8 h-8 text-green-600" />
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">Training Complete!</h4>
            <p className="text-gray-600 mb-6">
              Your model has been successfully trained and is ready for predictions
            </p>

            <div className="inline-flex items-center px-4 py-2 bg-green-100 text-green-800 rounded-lg">
              <CheckCircle className="w-4 h-4 mr-2" />
              Model Ready for Predictions
            </div>

            {/* Show download section */}
            <div className="mt-6">
              <ModelDownload isModelTrained={true} modelId={id!} />
            </div>
          </div>
)}


        {/* Training Configuration */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-3">Training Configuration</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Algorithm:</span>
              <span className="ml-2 font-medium text-gray-900">
                {algorithm.replace(/_/g, ' ').replace(/clf|reg/g, '').trim()}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Task Type:</span>
              <span className="ml-2 font-medium text-gray-900 capitalize">{taskType}</span>
            </div>
            <div>
              <span className="text-gray-600">Target Column:</span>
              <span className="ml-2 font-medium text-gray-900">{targetColumn}</span>
            </div>
            <div>
              <span className="text-gray-600">Features:</span>
              <span className="ml-2 font-medium text-gray-900">{featureColumns.length} columns</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}