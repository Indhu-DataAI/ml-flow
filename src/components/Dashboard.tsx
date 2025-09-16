import React, { useState, useEffect } from 'react';
import { ModelCard, Model } from './ModelCard';
// import { MetricsChart } from './MetricsChart';
import { Brain, Play, Download, Settings, RefreshCw, MailPlus } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import apiService from '../services/api';

interface AlgoMetrics {
  algorithm: string;
  [key: string]: any; // accuracy, precision, recall, rmse, etc.
}

interface ModelTrainingProps {
  taskType: 'classification' | 'regression';
  targetColumn: string;
  featureColumns: string[];
  onTrainComplete: (metrics: AlgoMetrics[]) => void;
}

interface TrainingResult {
  message: string;
  metrics: Record<string, any>; // algo_result from backend
  algorithm: string;
  task_type: string;
  training_samples: number;
  test_samples: number;
  features_used: number;
}

interface DashboardProps {
  taskType?: 'classification' | 'regression';
  targetColumn?: string;
  featureColumns?: string[];
  onTrainComplete?: (metrics: AlgoMetrics[]) => void;
}

export default function Dashboard({ taskType, targetColumn, featureColumns, onTrainComplete }: DashboardProps) {
  const location = useLocation();
  const state = location.state as any;
  
  const finalTaskType = taskType || state?.taskType || 'classification';
  const finalTargetColumn = targetColumn || state?.targetColumn || 'target';
  const finalFeatureColumns = featureColumns || state?.featureColumns || [];
  const finalOnTrainComplete = onTrainComplete || (() => {});
  const [isTraining, setIsTraining] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [trainingResults, setTrainingResults] = useState<TrainingResult | null>(null);

  // Initialize models with available algorithms
  useEffect(() => {
    const initializeModels = () => {
      const algorithmList = finalTaskType === 'classification' 
        ? [
            { id: 'random_forest_clf', name: 'Random Forest', type: 'Classification' },
            { id: 'svm_clf', name: 'Support Vector Machine', type: 'Classification' },
            { id: 'knn_clf', name: 'K-Nearest Neighbors', type: 'Classification' },
            { id: 'logistic_regression', name: 'Logistic Regression', type: 'Classification' },
            { id: 'decision_tree_clf', name: 'Decision Tree', type: 'Classification' },
            { id: 'gradient_boosting_clf', name: 'Gradient Boosting', type: 'Classification' },
            { id: 'naive_bayes', name: 'Naive Bayes', type: 'Classification' },
            { id: 'xgboost_clf', name: 'XGBoost', type: 'Classification' },
            {id:'mlp_clf',name:'MLP',type:'Classification'},
            {id:'ada_boost_clf',name:'AdaBoost',type:'Classification'}
          ]
        : [
            { id: 'random_forest_reg', name: 'Random Forest', type: 'Regression' },
            { id: 'gradient_boosting_reg', name: 'Gradient Boosting', type: 'Regression' },
            { id: 'linear_regression', name: 'Linear Regression', type: 'Regression' },
            {id:'xgboost_reg',name:'',type:'Classification'},
            { id: 'ridge_regression', name: 'Ridge Regression', type: 'Regression' },
            { id: 'lasso_regression', name: 'Lasso Regression', type: 'Regression' },
            { id: 'svm_reg', name: 'Support Vector Machine', type: 'Regression' },
            { id: 'decision_tree_reg', name: 'Decision Tree', type: 'Regression' },
            { id: 'knn_reg', name: 'K-Nearest Neighbors', type: 'Regression' },
            {id:'mlp_clf',name:'MLP',type:'Classification'},
            {id:'ada_boost_clf',name:'AdaBoost',type:'Classification'}      
      
           
          ];

      const initialModels: Model[] = algorithmList.map(algo => ({
        id: algo.id,
        name: algo.name,
        type: algo.type,
        status: 'idle' as const,
        progress: 0,
        metrics: undefined
      }));

      setModels(initialModels);
    };

    initializeModels();
  }, [finalTaskType]);

  const startTraining = async () => {
    setIsTraining(true);
    setError(null);
    setIsComplete(false);

    try {
      console.log('State data:', state);
      console.log('Dataset file:', state?.datasetFile);
      
      // Upload dataset first if available
      const datasetFile = state?.datasetFile;
      if (datasetFile) {
        const formData = new FormData();
        formData.append('file', datasetFile);
        const uploadResponse = await fetch('https://ml-platform-sl4g.onrender.com/upload-dataset', {
          method: 'POST',
          body: formData
        });
        
        if (!uploadResponse.ok) {
          throw new Error(`Dataset upload failed: ${uploadResponse.statusText}`);
        }
        
        console.log('Dataset uploaded successfully');
      } else {
        throw new Error('No dataset file available. Please go back and upload a dataset first.');
      }

      // Sequential training simulation
      const modelIds = models.map(m => m.id);
      let currentModelIndex = 0;
      
      const progressInterval = setInterval(() => {
        if (currentModelIndex < modelIds.length) {
          const currentModelId = modelIds[currentModelIndex];
          
          setModels(prevModels => 
            prevModels.map(model => {
              if (model.id === currentModelId) {
                const newProgress = Math.min(model.progress + Math.random() * 20, 90);
                if (newProgress >= 90) {
                  currentModelIndex++;
                  return { ...model, status: 'training' as const, progress: 100 };
                }
                return { ...model, status: 'training' as const, progress: newProgress };
              }
              return model;
            })
          );
        }
      }, 300);

      const result: TrainingResult = await apiService.trainAllModels({
        task_type: finalTaskType,
        target_column: finalTargetColumn,
        feature_columns: finalFeatureColumns,
      });
      
      clearInterval(progressInterval);
      
      setTrainingResults(result);
      
      // Convert backend metrics to AlgoMetrics array format
      const metricsArray: AlgoMetrics[] = Object.entries(result.metrics).map(([algorithm, metrics]) => ({
        algorithm,
        ...metrics
      }));

      // Update models with results
      setModels(prevModels => 
        prevModels.map(model => {
          const modelMetrics = result.metrics[model.id];
          if (modelMetrics) {
            return {
              ...model,
              status: 'completed' as const,
              progress: 100,
              metrics: {
                accuracy: parseFloat(modelMetrics.accuracy || modelMetrics.r2Score || '0'),
                loss: parseFloat(modelMetrics.rmse || modelMetrics.mae || '0'),
                precision: parseFloat(modelMetrics.precision || '0'),
                f1Score: parseFloat(modelMetrics.f1Score || '0'),
                trainingTime: Math.random() * 10 + 1, // Placeholder since not in backend
                ...modelMetrics
              }
            };
          }
          return { ...model, status: 'error' as const, progress: 0 };
        })
      );
      
      setIsTraining(false);
      setIsComplete(true);
      finalOnTrainComplete(metricsArray);
      
    } catch (err: any) {
      setIsTraining(false);
      console.error('Training error:', err);
      console.error('Response data:', err.response?.data);
      const errorMessage = err.response?.data?.detail || err.response?.data?.message || err.message || 'Training failed';
      setError(`Backend Error: ${JSON.stringify(err.response?.data) || err.message}`);
      
      // Update all models to error status
      setModels(prevModels => 
        prevModels.map(model => ({ ...model, status: 'error' as const, progress: 0 }))
      );
    }
  };

  const handleTrainAllModels = () => {
    startTraining();
  };

  const handleTrainModel = (modelId: string) => {
    // For individual model training, you might want to implement a separate endpoint
    console.log('Training individual model:', modelId);
  };
  const [downloading, setDownloading] = useState<string | null>(null);

  const downloadFile = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  const handleDownloadModel = async (modelId: string) => {
      setDownloading(modelId);
      try {
        const response = await apiService.downloadModel();
        const blob = new Blob([atob(response.model_data)], { type: 'application/octet-stream' });
        downloadFile(blob, response.filename);
      } catch (error: any) {
        console.error('Failed to download model:', error);
        setError(`Download failed: ${error.response?.data?.detail || error.message}`);
      } finally {
        setDownloading(null);
      }
    };
  

  // const handleResetAllModels = () => {
  //   setModels(prevModels => 
  //     prevModels.map(model => ({ 
  //       ...model, 
  //       status: 'idle' as const, 
  //       metrics: null 
  //     }))
  //   );
  //   setSelectedModel('');
  //   setTrainingResults(null);
  //   setIsComplete(false);
  //   setError(null);
  // };

  const trainingCount = models.filter(m => m.status === 'training').length;
  const completedCount = models.filter(m => m.status === 'completed').length;
  const isTrainingAll = isTraining;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <Brain className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">ML Model Training Dashboard</h1>
                <p className="text-sm text-gray-600">Train, compare, and download machine learning models</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-4 text-sm text-gray-600">
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  Training: {trainingCount}
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  Completed: {completedCount}
                </span>
              </div>
              
              {/* <button
                onClick={handleResetAllModels}
                className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors duration-200"
                title="Reset All Models"
              >
                <RefreshCw className="h-4 w-4" />
                Reset
              </button> */}
              
              <button
                onClick={handleTrainAllModels}
                disabled={isTrainingAll || trainingCount > 0}
                className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
              >
                <Play className="h-4 w-4" />
                {isTraining ? 'Training...' : 'Train All Models'}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        
        {trainingResults && (
          <div className="mb-8 p-6 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-900 mb-2">Training Complete</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-blue-700">Training Samples:</span>
                <div className="font-semibold text-blue-900">{trainingResults.training_samples}</div>
              </div>
              <div>
                <span className="text-blue-700">Test Samples:</span>
                <div className="font-semibold text-blue-900">{trainingResults.test_samples}</div>
              </div>
              <div>
                <span className="text-blue-700">Features Used:</span>
                <div className="font-semibold text-blue-900">{trainingResults.features_used}</div>
              </div>
              <div>
                <span className="text-blue-700">Task Type:</span>
                <div className="font-semibold text-blue-900 capitalize">{trainingResults.task_type}</div>
              </div>
            </div>
          </div>
        )}

       
        {/* Model Grid */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">Available Models</h2>
            {selectedModel && (
              <div className="flex items-center gap-2 text-sm text-blue-600">
                <Settings className="h-4 w-4" />
                <span>Model "{models.find(m => m.id === selectedModel)?.name}" selected</span>
                <button
                  onClick={() => setSelectedModel('')}
                  className="text-gray-400 hover:text-gray-600 ml-2"
                >
                  ×
                </button>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {models.map(model => (
              <ModelCard
                key={model.id}
                model={model}
                onTrain={handleTrainModel}
                onDownload={handleDownloadModel}
                onSelect={setSelectedModel}
                isSelected={selectedModel === model.id}
              />
            ))}
          </div>
        </div>

        {/* Selected Model Details */}
        {selectedModel && (
          <div className="bg-white p-6 rounded-xl border border-gray-200">
            {(() => {
              const model = models.find(m => m.id === selectedModel);
              if (!model) return null;

              return (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold text-gray-900">Selected Model Details</h3>
                    {model.status === 'completed' && (
                      <button
                        onClick={() => handleDownloadModel(model.id)}
                        disabled={downloading === model.id}
                        className="flex items-center gap-2 bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600 transition-colors duration-200"
                      >
                        <Download className="h-4 w-4" />
                        Download {model.name}
                      </button>


                    )}
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Model Information</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Name:</span>
                          <span className="font-medium">{model.name}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Type:</span>
                          <span className="font-medium">{model.type}</span>
                        </div>

                        <div className="flex justify-between">
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

                    {model.metrics && model.status === 'completed' && (
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-3">Performance Metrics</h4>
                        {finalTaskType === 'classification' ? (
                          <div className="grid grid-cols-2 gap-3">
                            <div className="text-center p-3 bg-blue-50 rounded-lg border border-blue-200">
                              <div className="text-xl font-bold text-blue-600">
                                {model.metrics.accuracy ? (parseFloat(String(model.metrics.accuracy)) * 100).toFixed(2) : 'N/A'}%
                              </div>
                              <div className="text-sm text-blue-700">Accuracy</div>
                            </div>
                            <div className="text-center p-3 bg-purple-50 rounded-lg border border-purple-200">
                              <div className="text-xl font-bold text-purple-600">
                                {model.metrics.precision ? (parseFloat(String(model.metrics.precision)) * 100).toFixed(2) : 'N/A'}%
                              </div>
                              <div className="text-sm text-purple-700">Precision</div>
                            </div>
                            <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
                              <div className="text-xl font-bold text-green-600">
                                {model.metrics.recall ? (parseFloat(String(model.metrics.recall)) * 100).toFixed(2) : 'N/A'}%
                              </div>
                              <div className="text-sm text-green-700">Recall</div>
                            </div>
                            <div className="text-center p-3 bg-orange-50 rounded-lg border border-orange-200">
                              <div className="text-xl font-bold text-orange-600">
                                {model.metrics.f1Score ? (parseFloat(String(model.metrics.f1Score)) * 100).toFixed(2) : 'N/A'}%
                              </div>
                              <div className="text-sm text-orange-700">F1 Score</div>
                            </div>
                          </div>
                        ) : (
                          <div className="grid grid-cols-2 gap-3">
                            <div className="text-center p-3 bg-blue-50 rounded-lg border border-blue-200">
                              <div className="text-xl font-bold text-blue-600">
                                {model.metrics.r2Score ? parseFloat(String(model.metrics.r2Score)).toFixed(3) : 'N/A'}
                              </div>
                              <div className="text-sm text-blue-700">R² Score</div>
                            </div>
                            <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
                              <div className="text-xl font-bold text-green-600">
                                {model.metrics.rmse ? parseFloat(String(model.metrics.rmse)).toFixed(4) : 'N/A'}
                              </div>
                              <div className="text-sm text-green-700">RMSE</div>
                            </div>
                            <div className="text-center p-3 bg-purple-50 rounded-lg border border-purple-200">
                              <div className="text-xl font-bold text-purple-600">
                                {model.metrics.mae ? parseFloat(String(model.metrics.mae)).toFixed(4) : 'N/A'}
                              </div>
                              <div className="text-sm text-purple-700">MAE</div>
                            </div>
                            <div className="text-center p-3 bg-orange-50 rounded-lg border border-orange-200">
                              <div className="text-xl font-bold text-orange-600">
                                {model.metrics.mse ? parseFloat(String(model.metrics.mse)).toFixed(5) : 'N/A'}
                              </div>
                              <div className="text-sm text-orange-700">MSE</div>
                            </div>
                          </div>
                        )}

                        {/* Cross Validation Results */}
                        {model.metrics.crossValidation && (
                          <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                            <div className="text-center">
                              <div className="text-lg font-bold text-gray-700">
                                {parseFloat(String(model.metrics.crossValidation.mean)).toFixed(3)} ± {parseFloat(String(model.metrics.crossValidation.std)).toFixed(3)}
                              </div>
                              <div className="text-sm text-gray-600">Cross Validation Score</div>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </div>
    </div>
  );
};