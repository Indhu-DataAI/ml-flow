import React from 'react';
import { Brain, Zap, Target, BarChart3, TreePine, Network, Layers, GitBranch, TrendingUp, Activity } from 'lucide-react';

interface Algorithm {
  id: string;
  name: string;
  description: string;
  icon: any;
  complexity: 'Low' | 'Medium' | 'High';
  speed: 'Fast' | 'Medium' | 'Slow';
  accuracy: 'Good' | 'Better' | 'Best';
}

interface AlgorithmSelectionProps {
  taskType: 'classification' | 'regression';
  selectedAlgorithm: string | null;
  onAlgorithmSelect: (algorithmId: string) => void;
}

export default function AlgorithmSelection({ taskType, selectedAlgorithm, onAlgorithmSelect }: AlgorithmSelectionProps) {
  const classificationAlgorithms: Algorithm[] = [
    {
      id: 'random_forest_clf',
      name: 'Random Forest',
      description: 'Ensemble method using multiple decision trees',
      icon: TreePine,
      complexity: 'Medium',
      speed: 'Fast',
      accuracy: 'Better',
    },
    {
      id: 'xgboost_clf',
      name: 'XGBoost',
      description: 'Gradient boosting framework for high performance',
      icon: Zap,
      complexity: 'High',
      speed: 'Medium',
      accuracy: 'Best',
    },
    {
      id: 'gradient_boosting_clf',
      name: 'Gradient Boosting',
      description: 'Sequential ensemble method with adaptive learning',
      icon: TrendingUp,
      complexity: 'High',
      speed: 'Medium',
      accuracy: 'Best',
    },
    {
      id: 'decision_tree_clf',
      name: 'Decision Tree',
      description: 'Simple tree-based classification model',
      icon: GitBranch,
      complexity: 'Low',
      speed: 'Fast',
      accuracy: 'Good',
    },
    {
      id: 'naive_bayes',
      name: 'Naive Bayes',
      description: 'Probabilistic classifier based on Bayes theorem',
      icon: Brain,
      complexity: 'Low',
      speed: 'Fast',
      accuracy: 'Good',
    },
    {
      id: 'knn_clf',
      name: 'K-Nearest Neighbors',
      description: 'Instance-based learning using nearest neighbors',
      icon: Network,
      complexity: 'Low',
      speed: 'Medium',
      accuracy: 'Good',
    },
    {
      id: 'mlp_clf',
      name: 'Neural Network (MLP)',
      description: 'Multi-layer perceptron for complex patterns',
      icon: Layers,
      complexity: 'High',
      speed: 'Medium',
      accuracy: 'Better',
    },
    {
      id: 'svm_clf',
      name: 'Support Vector Machine',
      description: 'Finds optimal decision boundary between classes',
      icon: Target,
      complexity: 'Medium',
      speed: 'Medium',
      accuracy: 'Better',
    },
    {
      id: 'logistic_regression',
      name: 'Logistic Regression',
      description: 'Linear approach for binary/multiclass classification',
      icon: BarChart3,
      complexity: 'Low',
      speed: 'Fast',
      accuracy: 'Good',
    },
    {
      id: 'ada_boost_clf',
      name: 'AdaBoost',
      description: 'Adaptive boosting with weak learners',
      icon: Activity,
      complexity: 'Medium',
      speed: 'Fast',
      accuracy: 'Better',
    },
  ];

  const regressionAlgorithms: Algorithm[] = [
    {
      id: 'random_forest_reg',
      name: 'Random Forest',
      description: 'Ensemble method using multiple decision trees',
      icon: TreePine,
      complexity: 'Medium',
      speed: 'Fast',
      accuracy: 'Better',
    },
    {
      id: 'xgboost_reg',
      name: 'XGBoost',
      description: 'Gradient boosting for numerical predictions',
      icon: Zap,
      complexity: 'High',
      speed: 'Medium',
      accuracy: 'Best',
    },
    {
      id: 'gradient_boosting_reg',
      name: 'Gradient Boosting',
      description: 'Sequential ensemble for regression tasks',
      icon: TrendingUp,
      complexity: 'High',
      speed: 'Medium',
      accuracy: 'Best',
    },
    {
      id: 'decision_tree_reg',
      name: 'Decision Tree',
      description: 'Simple tree-based regression model',
      icon: GitBranch,
      complexity: 'Low',
      speed: 'Fast',
      accuracy: 'Good',
    },
    {
      id: 'knn_reg',
      name: 'K-Nearest Neighbors',
      description: 'Instance-based regression using nearest neighbors',
      icon: Network,
      complexity: 'Low',
      speed: 'Medium',
      accuracy: 'Good',
    },
    {
      id: 'mlp_reg',
      name: 'Neural Network (MLP)',
      description: 'Multi-layer perceptron for complex regression',
      icon: Layers,
      complexity: 'High',
      speed: 'Medium',
      accuracy: 'Better',
    },
    {
      id: 'linear_regression',
      name: 'Linear Regression',
      description: 'Simple linear relationship modeling',
      icon: BarChart3,
      complexity: 'Low',
      speed: 'Fast',
      accuracy: 'Good',
    },
    {
      id: 'ridge_regression',
      name: 'Ridge Regression',
      description: 'Linear regression with L2 regularization',
      icon: BarChart3,
      complexity: 'Low',
      speed: 'Fast',
      accuracy: 'Good',
    },
    {
      id: 'lasso_regression',
      name: 'Lasso Regression',
      description: 'Linear regression with L1 regularization',
      icon: BarChart3,
      complexity: 'Low',
      speed: 'Fast',
      accuracy: 'Good',
    },
    {
      id: 'svm_reg',
      name: 'Support Vector Regression',
      description: 'SVM approach for continuous value prediction',
      icon: Target,
      complexity: 'Medium',
      speed: 'Medium',
      accuracy: 'Better',
    },
    {
      id: 'ada_boost_reg',
      name: 'AdaBoost',
      description: 'Adaptive boosting for regression tasks',
      icon: Activity,
      complexity: 'Medium',
      speed: 'Fast',
      accuracy: 'Better',
    },
  ];

  const algorithms = taskType === 'classification' ? classificationAlgorithms : regressionAlgorithms;

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'Low': return 'text-green-700 bg-green-100';
      case 'Medium': return 'text-yellow-700 bg-yellow-100';
      case 'High': return 'text-red-700 bg-red-100';
      default: return 'text-gray-700 bg-gray-100';
    }
  };

  const getSpeedColor = (speed: string) => {
    switch (speed) {
      case 'Fast': return 'text-green-700 bg-green-100';
      case 'Medium': return 'text-yellow-700 bg-yellow-100';
      case 'Slow': return 'text-red-700 bg-red-100';
      default: return 'text-gray-700 bg-gray-100';
    }
  };

  const getAccuracyColor = (accuracy: string) => {
    switch (accuracy) {
      case 'Good': return 'text-blue-700 bg-blue-100';
      case 'Better': return 'text-purple-700 bg-purple-100';
      case 'Best': return 'text-green-700 bg-green-100';
      default: return 'text-gray-700 bg-gray-100';
    }
  };

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Select Algorithm</h3>
        <p className="text-gray-600">
          Choose the machine learning algorithm for your {taskType} task
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {algorithms.map((algorithm) => {
          const Icon = algorithm.icon;
          const isSelected = selectedAlgorithm === algorithm.id;
          
          return (
            <div
              key={algorithm.id}
              onClick={() => onAlgorithmSelect(algorithm.id)}
              className={`p-6 rounded-xl border-2 cursor-pointer transition-all hover:shadow-md ${
                isSelected
                  ? 'border-blue-500 bg-blue-50 shadow-md'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Icon className="w-6 h-6 text-blue-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-gray-900 mb-2">{algorithm.name}</h4>
                  <p className="text-gray-600 text-sm mb-4">{algorithm.description}</p>
                  
                  <div className="flex flex-wrap gap-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getComplexityColor(algorithm.complexity)}`}>
                      {algorithm.complexity} Complexity
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSpeedColor(algorithm.speed)}`}>
                      {algorithm.speed} Speed
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getAccuracyColor(algorithm.accuracy)}`}>
                      {algorithm.accuracy} Accuracy
                    </span>
                  </div>
                </div>
              </div>
              
              {isSelected && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    <Brain className="w-3 h-3 mr-1" />
                    Selected
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}