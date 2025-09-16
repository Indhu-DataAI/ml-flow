import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Brain, TrendingUp } from 'lucide-react';
import apiService from '../services/api';

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface ExplainabilityData {
  feature_importance: FeatureImportance[];
  model_type: string;
  explanation_method: string;
}

interface ExplainabilityDashboardProps {
  isModelTrained: boolean;
}

export default function ExplainabilityDashboard({ isModelTrained }: ExplainabilityDashboardProps) {
  const [explainabilityData, setExplainabilityData] = useState<ExplainabilityData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isModelTrained) {
      fetchExplainability();
    }
  }, [isModelTrained]);

  const fetchExplainability = async () => {
    setLoading(true);
    try {
      const data = await apiService.getModelExplainability();
      setExplainabilityData(data);
    } catch (error) {
      console.error('Failed to fetch explainability:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!isModelTrained) {
    return null;
  }

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
          <Brain className="w-5 h-5 text-indigo-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Model Explainability</h3>
          <p className="text-sm text-gray-600">
            {explainabilityData ? `${explainabilityData.model_type} - ${explainabilityData.explanation_method}` : 'Feature importance analysis'}
          </p>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      ) : explainabilityData ? (
        <div className="space-y-6">
          {/* Feature Importance Chart */}
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={explainabilityData.feature_importance.slice(0, 10)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="feature" 
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  fontSize={12}
                />
                <YAxis fontSize={12} />
                <Tooltip />
                <Bar dataKey="importance" fill="#4f46e5" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Top Features List */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-900 mb-3 flex items-center">
              <TrendingUp className="w-4 h-4 mr-2" />
              Top Important Features
            </h4>
            <div className="space-y-2">
              {explainabilityData.feature_importance.slice(0, 5).map((item, index) => (
                <div key={item.feature} className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">
                    {index + 1}. {item.feature}
                  </span>
                  <span className="text-sm text-gray-600">
                    {(item.importance * 100).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          Failed to load explainability data
        </div>
      )}
    </div>
  );
}