import React from 'react';
import { Upload, Database, Target, Brain, BarChart3, Calculator } from 'lucide-react';

interface StepIndicatorProps {
  currentStep: number;
}

export default function StepIndicator({ currentStep }: StepIndicatorProps) {
  const steps = [
    { id: 1, name: 'Upload Data', icon: Upload },
    { id: 2, name: 'Preview & Configure', icon: Database },
    { id: 3, name: 'Select Task', icon: Target },
    { id: 4, name: 'Choose Algorithm', icon: Brain },
    { id: 5, name: 'Train Model', icon: BarChart3 },
    { id: 6, name: 'Make Predictions', icon: Calculator },
  ];

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 mb-8">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => {
          const Icon = step.icon;
          const isActive = currentStep === step.id;
          const isCompleted = currentStep > step.id;
          
          return (
            <React.Fragment key={step.id}>
              <div className="flex flex-col items-center">
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${
                    isCompleted
                      ? 'bg-green-600 border-green-600 text-white'
                      : isActive
                      ? 'bg-blue-600 border-blue-600 text-white'
                      : 'bg-white border-gray-300 text-gray-400'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                </div>
                <span
                  className={`text-sm mt-2 font-medium ${
                    isCompleted || isActive ? 'text-gray-900' : 'text-gray-500'
                  }`}
                >
                  {step.name}
                </span>
              </div>
              
              {index < steps.length - 1 && (
                <div
                  className={`flex-1 h-0.5 mx-4 ${
                    currentStep > step.id ? 'bg-green-600' : 'bg-gray-300'
                  }`}
                />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}