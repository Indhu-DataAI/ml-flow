import React from 'react';
import { BarChart3, Target, TrendingUp, CheckCircle } from 'lucide-react';

interface ClassificationMetrics {
  accuracy: string;
  precision: string;
  recall: string;
  f1Score: string;
  confusionMatrix: number[][];
}

interface RegressionMetrics {
  rmse: string;
  mae: string;
  r2Score: string;
  mse: string;
}

interface MetricsDashboardProps {
  metrics: ClassificationMetrics | RegressionMetrics;
  taskType: 'classification' | 'regression';
  algorithm: string;
}

export default function MetricsDashboard({ metrics, taskType, algorithm }: MetricsDashboardProps) {
  const isClassification = taskType === 'classification';
  const classificationMetrics = metrics as ClassificationMetrics;
  const regressionMetrics = metrics as RegressionMetrics;

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
          <CheckCircle className="w-6 h-6 text-green-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Model Performance</h3>
          <p className="text-gray-600">
            {algorithm.replace(/_/g, ' ').replace(/clf|reg/g, '').trim()} - {taskType}
          </p>
        </div>
      </div>

      {isClassification ? (
        <div className="space-y-6">
          {/* Classification Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <div className="flex items-center space-x-2 mb-2">
                <Target className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-blue-900">Accuracy</span>
              </div>
              <div className="text-2xl font-bold text-blue-900">{parseFloat(classificationMetrics.accuracy) > 1 ? parseFloat(classificationMetrics.accuracy).toFixed(1) + '%' : (parseFloat(classificationMetrics.accuracy) * 100).toFixed(1) + '%'}</div>
            </div>
            
            <div className="bg-green-50 rounded-lg p-4 border border-green-200">
              <div className="flex items-center space-x-2 mb-2">
                <BarChart3 className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-green-900">Precision</span>
              </div>
              <div className="text-2xl font-bold text-green-900">{(parseFloat(classificationMetrics.precision) * 100).toFixed(1)}%</div>
            </div>
            
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="w-4 h-4 text-purple-600" />
                <span className="text-sm font-medium text-purple-900">Recall</span>
              </div>
              <div className="text-2xl font-bold text-purple-900">{(parseFloat(classificationMetrics.recall) * 100).toFixed(1)}%</div>
            </div>
            
            <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
              <div className="flex items-center space-x-2 mb-2">
                <Target className="w-4 h-4 text-orange-600" />
                <span className="text-sm font-medium text-orange-900">F1-Score</span>
              </div>
              <div className="text-2xl font-bold text-orange-900">{(parseFloat(classificationMetrics.f1Score) * 100).toFixed(1)}%</div>
            </div>
          </div>

          {/* Confusion Matrix */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-900 mb-4">Confusion Matrix</h4>
            <div 
              className="grid gap-2 max-w-md"
              style={{ gridTemplateColumns: `repeat(${classificationMetrics.confusionMatrix.length}, 1fr)` }}
            >
              {classificationMetrics.confusionMatrix.map((row, i) =>
                row.map((value, j) => (
                  <div
                    key={`${i}-${j}`}
                    className={`p-4 rounded text-center font-bold ${
                      i === j
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {value}
                  </div>
                ))
              )}
            </div>
            <div className="mt-4 text-xs text-gray-600">
              <div className="mb-2"><strong>Predicted →</strong></div>
              <div className="grid gap-1">
                {classificationMetrics.confusionMatrix.map((row, i) => (
                  <div key={i}>
                    <strong>Class {i}:</strong> {row.map((val, j) => `${val} (→Class ${j})`).join(', ')}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Regression Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <div className="flex items-center space-x-2 mb-2">
                <BarChart3 className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-blue-900">RMSE</span>
              </div>
              <div className="text-2xl font-bold text-blue-900">{parseFloat(regressionMetrics.rmse).toFixed(4)}</div>
            </div>
            
            <div className="bg-green-50 rounded-lg p-4 border border-green-200">
              <div className="flex items-center space-x-2 mb-2">
                <Target className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-green-900">MAE</span>
              </div>
              <div className="text-2xl font-bold text-green-900">{parseFloat(regressionMetrics.mae).toFixed(4)}</div>
            </div>
            
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="w-4 h-4 text-purple-600" />
                <span className="text-sm font-medium text-purple-900">R² Score</span>
              </div>
              <div className="text-2xl font-bold text-purple-900">{(parseFloat(regressionMetrics.r2Score) * 100).toFixed(1)}%</div>
            </div>
            
            <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
              <div className="flex items-center space-x-2 mb-2">
                <BarChart3 className="w-4 h-4 text-orange-600" />
                <span className="text-sm font-medium text-orange-900">MSE</span>
              </div>
              <div className="text-2xl font-bold text-orange-900">{parseFloat(regressionMetrics.mse).toFixed(5)}</div>
            </div>
          </div>

          {/* Metric Explanations */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-900 mb-4">Metric Explanations</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <strong className="text-gray-900">RMSE (Root Mean Square Error):</strong>
                <p className="text-gray-600">Lower values indicate better model performance</p>
              </div>
              <div>
                <strong className="text-gray-900">MAE (Mean Absolute Error):</strong>
                <p className="text-gray-600">Average absolute difference between predictions and actual values</p>
              </div>
              <div>
                <strong className="text-gray-900">R² Score:</strong>
                <p className="text-gray-600">Proportion of variance explained by the model (higher is better)</p>
              </div>
              <div>
                <strong className="text-gray-900">MSE (Mean Square Error):</strong>
                <p className="text-gray-600">Average of squared differences between predictions and actual values</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}