import React, { useState } from 'react';
import { Calculator, Zap, ArrowRight, AlertCircle } from 'lucide-react';
import apiService from '../services/api';

interface Column {
  name: string;
  type: string;
  included: boolean;
  isTarget: boolean;
}

interface PredictionPanelProps {
  columns: Column[];
  taskType: 'classification' | 'regression';
  algorithm: string;
}

export default function PredictionPanel({ columns, taskType, algorithm }: PredictionPanelProps) {
  const [inputValues, setInputValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const featureColumns = columns.filter(col => col.included && !col.isTarget);

  const handleInputChange = (columnName: string, value: string) => {
    setInputValues(prev => ({
      ...prev,
      [columnName]: value
    }));
    setError(null);
  };

  const validateInputs = () => {
    for (const column of featureColumns) {
      if (!inputValues[column.name] || inputValues[column.name].trim() === '') {
        return `Please provide a value for ${column.name}`;
      }
    }
    return null;
  };

  const handlePredict = async (withExplanation = false) => {
    const validationError = validateInputs();
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Convert string inputs to appropriate types
      const processedInputs: Record<string, string | number> = {};
      featureColumns.forEach(column => {
        const value = inputValues[column.name];
        if (column.type === 'number') {
          processedInputs[column.name] = parseFloat(value);
        } else {
          processedInputs[column.name] = value;
        }
      });

      const result = await apiService.predict({ 
        input_data: processedInputs, 
       
      });
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to make prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setInputValues({});
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
          <Calculator className="w-6 h-6 text-blue-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Make Predictions</h3>
          <p className="text-gray-600">Enter values to get predictions from your trained model</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="space-y-4">
          <h4 className="font-medium text-gray-900 mb-4">Input Features</h4>
          
          {featureColumns.map((column) => (
            <div key={column.name}>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {column.name}
                <span className="text-gray-500 ml-1">({column.type})</span>
              </label>
              <input
                type={column.type === 'number' ? 'number' : 'text'}
                value={inputValues[column.name] || ''}
                onChange={(e) => handleInputChange(column.name, e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder={`Enter ${column.name}`}
              />
            </div>
          ))}

          {error && (
            <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-lg">
              <AlertCircle className="w-4 h-4 text-red-600" />
              <span className="text-sm text-red-700">{error}</span>
            </div>
          )}

          <div className="space-y-3 pt-4">
            <div >
              <button
                onClick={() => handlePredict(false)}
                disabled={isLoading}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-400 transition-colors"
              >
                {isLoading ? (
                  <>
                    <Zap className="w-4 h-4 mr-2 animate-spin" />
                    Predicting...
                  </>
                ) : (
                  <>
                   
                    Predict
                  </>
                )}
              </button>
              
              
            </div>
            
            <button
              onClick={resetForm}
              className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Prediction Results */}
        <div>
          <h4 className="font-medium text-gray-900 mb-4">Prediction Results</h4>
          
          {prediction ? (
            <div className="space-y-4">
              {taskType === 'classification' ? (
                <div>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Zap className="w-5 h-5 text-green-600" />
                      <span className="font-semibold text-green-900">Predicted Class</span>
                    </div>
                    <div className="text-2xl font-bold text-green-900 mb-1">
                      {prediction.predictedClass}
                    </div>
                    <div className="text-sm text-green-700">
                      Confidence: {prediction.confidence}%
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-3">Class Probabilities</h5>
                    <div className="space-y-2">
                      {prediction.probabilities.map((item: any, index: number) => (
                        <div key={index} className="flex items-center space-x-3">
                          <span className="text-sm font-medium text-gray-700 w-16">
                            {item.class}
                          </span>
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${item.probability}%` }}
                            />
                          </div>
                          <span className="text-sm text-gray-600 w-12">
                            {item.probability}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Calculator className="w-5 h-5 text-blue-600" />
                    <span className="font-semibold text-blue-900">Predicted Value</span>
                  </div>
                  <div className="text-3xl font-bold text-blue-900 mb-1">
                    {prediction.predictedValue}
                  </div>
                  <div className="text-sm text-blue-700">
                    Confidence: {prediction.confidence}%
                  </div>
                </div>
              )}
              
              {/* Feature Impact */}
              {prediction.feature_impact && (
                <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                  <h5 className="font-medium text-purple-900 mb-3">Feature Impact</h5>
                  <div className="space-y-2">
                    {prediction.feature_impact.slice(0, 5).map((impact: any, index: number) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-sm font-medium text-purple-800">{impact.feature}</span>
                        <div className="flex items-center space-x-2">
                          <div 
                            className={`h-2 rounded ${
                              impact.impact > 0 ? 'bg-green-400' : 'bg-red-400'
                            }`}
                            style={{ width: `${Math.min(Math.abs(impact.impact) * 50, 50)}px` }}
                          />
                          <span className="text-xs text-purple-600 w-16 text-right">
                            {impact.impact > 0 ? '+' : ''}{impact.impact.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-purple-600 mt-2">
                    Green: increases prediction, Red: decreases prediction
                  </p>
                </div>
              )}
              
              <div className="bg-gray-100 rounded-lg p-4">
                <h5 className="font-medium text-gray-900 mb-2">Model Information</h5>
                <div className="text-sm text-gray-600 space-y-1">
                  <div>Algorithm: {algorithm.replace(/_/g, ' ').replace(/clf|reg/g, '').trim()}</div>
                  <div>Task Type: {taskType}</div>
                  <div>Features Used: {featureColumns.length}</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-gray-50 rounded-lg p-8 text-center">
              <Calculator className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-600">
                Fill in the input values and click "Predict" to see results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}