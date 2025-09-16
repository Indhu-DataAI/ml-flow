import React from 'react';
import { BarChart3, TrendingUp, Brain, Target } from 'lucide-react';

interface TaskSelectionProps {
  selectedTask: 'classification' | 'regression' | null;
  onTaskSelect: (task: 'classification' | 'regression') => void;
  targetColumn: string | null;
}

export default function TaskSelection({ selectedTask, onTaskSelect, targetColumn }: TaskSelectionProps) {
  const tasks = [
    {
      id: 'classification' as const,
      title: 'Classification',
      description: 'Predict categorical outcomes or classes',
      icon: Target,
      examples: ['Email spam detection', 'Image recognition', 'Customer churn'],
      color: 'blue',
    },
    {
      id: 'regression' as const,
      title: 'Regression',
      description: 'Predict continuous numerical values',
      icon: TrendingUp,
      examples: ['House price prediction', 'Sales forecasting', 'Stock prices'],
      color: 'green',
    },
  ];

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Select Task Type</h3>
        <p className="text-gray-600">
          Choose the type of machine learning task based on your target variable
          {targetColumn && <span className="font-medium ml-1">"{targetColumn}"</span>}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {tasks.map((task) => {
          const Icon = task.icon;
          const isSelected = selectedTask === task.id;
          
          return (
            <div
              key={task.id}
              onClick={() => onTaskSelect(task.id)}
              className={`p-6 rounded-xl border-2 cursor-pointer transition-all hover:shadow-md ${
                isSelected
                  ? task.color === 'blue'
                    ? 'border-blue-500 bg-blue-50 shadow-md'
                    : 'border-green-500 bg-green-50 shadow-md'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-start space-x-4">
                <div
                  className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                    task.color === 'blue' ? 'bg-blue-100' : 'bg-green-100'
                  }`}
                >
                  <Icon
                    className={`w-6 h-6 ${
                      task.color === 'blue' ? 'text-blue-600' : 'text-green-600'
                    }`}
                  />
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-gray-900 mb-2">{task.title}</h4>
                  <p className="text-gray-600 text-sm mb-4">{task.description}</p>
                  
                  <div>
                    <p className="text-xs font-medium text-gray-700 mb-2">Use cases:</p>
                    <ul className="space-y-1">
                      {task.examples.map((example, index) => (
                        <li key={index} className="text-xs text-gray-600 flex items-center">
                          <div className="w-1 h-1 bg-gray-400 rounded-full mr-2" />
                          {example}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
              
              {isSelected && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div
                    className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                      task.color === 'blue'
                        ? 'bg-blue-100 text-blue-800'
                        : 'bg-green-100 text-green-800'
                    }`}
                  >
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