import axios from 'axios';

const API_BASE_URL = 'https://ml-platform-sl4g.onrender.com';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});
export interface TrainAllModelsRequest {
  task_type: 'classification' | 'regression';
  target_column: string;
  feature_columns: string[];
}

export interface ModelMetrics {
  accuracy?: string;
  precision?: string;
  recall?: string;
  f1Score?: string;
  rmse?: string;
  mae?: string;
  r2Score?: string;
  mse?: string;
}

export interface TrainAllModelsResponse {
  message: string;
  metrics: { [algorithm: string]: ModelMetrics };
  algorithm: string;
  task_type: string;
  training_samples: number;
  test_samples: number;
  features_used: number;
}


export interface TrainingRequest {
  algorithm: string;
  task_type: string;
  target_column: string;
  feature_columns: string[];
}

export interface PredictionRequest {
  input_data: Record<string, string | number>;

}


export const apiService = {
  // Upload dataset
  uploadDataset: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/upload-dataset', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Train model
  trainModel: async (request: TrainingRequest) => {
    const response = await api.post('/train-model', request);
    return response.data;
  },
  trainAllModels: async(request: TrainAllModelsRequest)=> {
    const response = await api.post('/train-all-models', request);
    return response.data;
},

  // Make prediction
  predict: async (request: PredictionRequest) => {
    const response = await api.post('/predict', request);
    return response.data;
  },

  // Download model
  downloadModel: async (modelId?: string) => {
    const url = modelId ? `/download-model/${modelId}` : '/download-model';
    const response = await api.get(url, {
      responseType: 'blob'
    });
    const filename = response.headers['content-disposition']?.split('filename=')[1] || 'model.joblib';
    return { blob: response.data, filename };
  },

  // Download encoders
  downloadEncoders: async () => {
    const response = await api.get('/download-encoders', {
      responseType: 'blob'
    });
    const filename = response.headers['content-disposition']?.split('filename=')[1] || 'encoders.pkl';
    return { blob: response.data, filename };
  },

  // Get model explainability
  getModelExplainability: async () => {
    const response = await api.get('/model-explainability');
    return response.data;
  },

  // Get model status
  getModelStatus: async () => {
    const response = await api.get('/model-status');
    return response.data;
  },
};

export default apiService;

