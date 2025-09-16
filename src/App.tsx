import React, { useState } from 'react';
import { Brain } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import apiService from './services/api';
import FileUpload from './components/FileUpload';
import DataPreview from './components/DataPreview';
import TaskSelection from './components/TaskSelection';
import AlgorithmSelection from './components/AlgorithmSelection';
import ModelTraining from './components/ModelTraining';
import MetricsDashboard from './components/MetricsDashboard';
import PredictionPanel from './components/PredictionPanel';
import StepIndicator from './components/StepIndicator';
import ModelDownload from './components/ModelDownload';
import ExplainabilityDashboard from './components/ExplainabilityDashboard';
import logo from './logo.png';
import Dashboard from './components/Dashboard';

interface Column {
  name: string;
  type: string;
  included: boolean;
  isTarget: boolean;
  removed: boolean;
}

function App() {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [data, setData] = useState<any[]>([]);
  const [columns, setColumns] = useState<Column[]>([]);
  const [selectedTask, setSelectedTask] = useState<'classification' | 'regression' | null>(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [isModelTrained, setIsModelTrained] = useState(false);

  const handleFileSelect = (file: File, uploadedData: any[], uploadedColumns: any[]) => {
    setSelectedFile(file);
    setData(uploadedData);
    setColumns(uploadedColumns);
    setCurrentStep(2);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setData([]);
    setColumns([]);
    setSelectedTask(null);
    setSelectedAlgorithm(null);
    setMetrics(null);
    setIsModelTrained(false);
    setCurrentStep(1);
  };

  const handleToggleColumn = (columnName: string) => {
    setColumns(prev => prev.map(col => 
      col.name === columnName ? { ...col, included: !col.included } : col
    ));
  };

  const handleSetTarget = (columnName: string) => {
    setColumns(prev => prev.map(col => ({
      ...col,
      isTarget: col.name === columnName ? !col.isTarget : false,
    })));
  };

  const handleRemoveColumn = (columnName: string) => {
    setColumns(prev => prev.map(col => 
      col.name === columnName ? { ...col, removed: !col.removed, included: false, isTarget: false } : col
    ));
  };
  const handleTaskSelect = (task: 'classification' | 'regression') => {
    setSelectedTask(task);
    setCurrentStep(4);
  };

  const navigate = useNavigate();

const handleGoToDashboard = () => {
  if (selectedTask && targetColumn) {
    navigate("/dashboard", {
      state: {
        taskType: selectedTask,
        targetColumn: targetColumn.name,
        featureColumns: featureColumns,
        datasetFile: selectedFile,
        data: data,
        columns: columns
      },
    });
  }
};


  const handleAlgorithmSelect = (algorithmId: string) => {
    setSelectedAlgorithm(algorithmId);
    setCurrentStep(5);
  };

  const handleTrainComplete = (trainingMetrics: any) => {
    setMetrics(trainingMetrics);
    setIsModelTrained(true);
    setCurrentStep(6);
  };

  const canProceedToTaskSelection = () => {
    return columns.some(col => col.isTarget && !col.removed) && 
           columns.some(col => col.included && !col.isTarget && !col.removed);
  };

  const targetColumn = columns.find(col => col.isTarget && !col.removed);
  const featureColumns = columns.filter(col => col.included && !col.isTarget && !col.removed).map(col => col.name);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-20 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                 <img
                    src={logo}
                    alt="Logo"
                    className="w-68 h-8"
                  />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">BCT ML Studio</h1>
                <p className="text-sm text-gray-500">Machine Learning Platform</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <StepIndicator currentStep={currentStep} />

        <div className="space-y-8">
          {/* Step 1: File Upload */}
          {currentStep >= 1 && (
            <FileUpload
              onFileSelect={handleFileSelect}
              selectedFile={selectedFile}
              onRemoveFile={handleRemoveFile}
            />
          )}

          {/* Step 2: Data Preview */}
          {currentStep >= 2 && data.length > 0 && (
            <div className="space-y-6">
              <DataPreview
                data={data}
                columns={columns}
                onToggleColumn={handleToggleColumn}
                onSetTarget={handleSetTarget}
                onRemoveColumn={handleRemoveColumn}
              />
              
              {canProceedToTaskSelection() && currentStep === 2 && (
                <div className="text-center">
                  <button
                    onClick={() => setCurrentStep(3)}
                    className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Continue to Task Selection
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Step 3: Task Selection */}
          {currentStep >= 3 && targetColumn && (
            <div className="space-y-6">
              <TaskSelection
                selectedTask={selectedTask}
                onTaskSelect={handleTaskSelect}
                targetColumn={targetColumn.name}
              />
              
              {selectedTask && (
                <div className="text-center">
                  <button
                    onClick={handleGoToDashboard}
                    className="inline-flex items-center px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    Go to Training Dashboard
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Step 4: Algorithm Selection */}
          {currentStep >= 4 && selectedTask && (
            <AlgorithmSelection
              taskType={selectedTask}
              selectedAlgorithm={selectedAlgorithm}
              onAlgorithmSelect={handleAlgorithmSelect}
            />
          )}

          {/* Step 5: Model Training */}
          {currentStep >= 5 && selectedAlgorithm && (
            <ModelTraining
              algorithm={selectedAlgorithm}
              taskType={selectedTask!}
              targetColumn={targetColumn!.name}
              featureColumns={featureColumns}
              onTrainComplete={handleTrainComplete}
            />
          )}

          {/* {currentStep >= 5 && selectedAlgorithm && (
            <Dashboard
              taskType={selectedTask!}
              targetColumn={targetColumn!.name}
              featureColumns={featureColumns}
              onTrainComplete={handleTrainComplete}
            />
          )}
           */}

          {/* Step 6: Results */}
          {currentStep >= 6 && metrics && selectedTask && selectedAlgorithm && (
            <div className="space-y-8">
              <MetricsDashboard
                metrics={metrics}
                taskType={selectedTask}
                algorithm={selectedAlgorithm}
              />
              
              <ExplainabilityDashboard isModelTrained={isModelTrained} />
              
              
              
              <PredictionPanel
                columns={columns}
                taskType={selectedTask}
                algorithm={selectedAlgorithm}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;