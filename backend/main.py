from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import uuid
import io
import joblib
import logging
from typing import Dict, Any, List, Optional, Union
import warnings
import base64
from sklearn.inspection import permutation_importance
# import shap
from fastapi.responses import StreamingResponse

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Studio API",
    version="1.0.0",
    description="Advanced Machine Learning API with comprehensive algorithm support"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model state
current_model = None
current_scaler = None
current_label_encoder = None
current_dataset = None
feature_columns = []
target_column = ""
task_type = ""
model_metrics = {}
feature_encoders = {} 
model_explainer = None
X_train_sample = None
algo_result={}
# Store trained models by ID
trained_models = {}
trained_models = {}  # NEW: store multiple models


class TrainingRequest(BaseModel):
    algorithm: str = Field(..., description="Algorithm name")
    task_type: str = Field(..., description="'classification' or 'regression'")
    target_column: str = Field(..., description="Target column name")
    feature_columns: List[str] = Field(..., description="List of feature column names")
class PredictionRequest(BaseModel):
    input_data: Dict[str, Union[int, float, str]] = Field(..., description="Input features for prediction")
class ModelConfig(BaseModel):
    algorithm: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
class Trainingall(BaseModel):
    task_type: str = Field(..., description="'classification' or 'regression'")
    target_column: str = Field(..., description="Target column name")
    feature_columns: List[str] = Field(..., description="List of feature column names")

# Algorithm mappings
CLASSIFICATION_ALGORITHMS = {
    "random_forest_clf": lambda: RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
),
"svm_clf": lambda: SVC(
    probability=True, random_state=42, class_weight="balanced" 
),
"knn_clf": lambda: KNeighborsClassifier(n_neighbors=5),
"logistic_regression": lambda: LogisticRegression(
    random_state=42, max_iter=1000, class_weight="balanced" 
),

"decision_tree_clf": lambda: DecisionTreeClassifier(
    random_state=42, class_weight="balanced"
),

"gradient_boosting_clf": lambda: GradientBoostingClassifier(
    n_estimators=100, random_state=42  
),

"naive_bayes": lambda: GaussianNB(),
"xgboost_clf": lambda: XGBClassifier(
    n_estimators=100,
    use_label_encoder=False, eval_metric='logloss', random_state=42,
    scale_pos_weight=1  
),

"mlp_clf": lambda: MLPClassifier(
    hidden_layer_sizes=(100,), max_iter=500, random_state=42),
"ada_boost_clf": lambda: AdaBoostClassifier(n_estimators=50, random_state=42)
}

REGRESSION_ALGORITHMS = {
    "random_forest_reg": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting_reg": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "linear_regression": lambda: LinearRegression(),
    "xgboost_reg": lambda: XGBRegressor(n_estimators=100,random_state=42),
    "ridge_regression": lambda: Ridge(alpha=1.0, random_state=42),
    "lasso_regression": lambda: Lasso(alpha=1.0, random_state=42, max_iter=1000),
    "svm_reg": lambda: SVR(),
    "decision_tree_reg": lambda: DecisionTreeRegressor(random_state=42),
    "knn_reg": lambda: KNeighborsRegressor(n_neighbors=5),
    "mlp_reg": lambda: MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "ada_boost_reg": lambda: AdaBoostRegressor(n_estimators=50, random_state=42),
}

# Algorithms that require feature scaling
SCALING_ALGORITHMS = {
    'logistic_regression', 'svm_clf', 'svm_reg', 'linear_regression',
    'ridge_regression', 'lasso_regression', 'mlp_clf', 'mlp_reg', 'knn_clf', 'knn_reg'
}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

@app.get("/")
async def root():
    return {
        "message": "ML Flow API v.0 is running"
        
    }

@app.get("/algorithms")
async def get_algorithms():
    """Get list of available algorithms"""
    return {
        "classification": list(CLASSIFICATION_ALGORITHMS.keys()),
        "regression": list(REGRESSION_ALGORITHMS.keys()),
        "total": len(CLASSIFICATION_ALGORITHMS) + len(REGRESSION_ALGORITHMS)
    }
UPLOAD_PATH = os.path.join(os.getcwd(), "uploaded_dataset.csv")


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and analyze CSV dataset"""
    global current_dataset
    
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        df.to_csv(UPLOAD_PATH, index=False)  # persist
        current_dataset = df
        logger.info(f"Dataset uploaded: {df.shape[0]} rows, {df.shape[1]} columns")
        columns_info = []
        for col in df.columns:
            col_info = {
                "name": col,
                "type": "number" if pd.api.types.is_numeric_dtype(df[col]) else "string",
                "missing_values": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique()),
                "included": True,
                "isTarget": False
            }
            if col_info["type"] == "number":
                col_info["sample_values"] = df[col].dropna().head(3).tolist()
                col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
            else:
                col_info["sample_values"] = df[col].dropna().head(3).tolist()
            
            columns_info.append(col_info)
        
        # Dataset statistics
        dataset_stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        
        # Return preview data (first 100 rows)
        preview_data = df.head(10).fillna("").to_dict('records')
        
        return {    
            "data": preview_data,
            "columns": columns_info,
            "stats": dataset_stats,
            "message": "Dataset uploaded and analyzed successfully"
        }
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty or corrupted")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/train-model")
async def train_model(request: TrainingRequest):
    """Train machine learning model with comprehensive error handling"""
    global current_model, current_scaler, current_label_encoder, current_dataset
    global feature_columns, target_column, task_type, model_metrics
    global feature_encoders
    feature_encoders = {}
    
    try:
        # Validate dataset exists
        if current_dataset is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")
        
        df = current_dataset.copy()
        
        # Validate columns exist
        missing_features = [col for col in request.feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Feature columns not found in dataset: {missing_features}"
            )
        
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in dataset"
            )
        
        # Validate algorithm
        all_algorithms = {**CLASSIFICATION_ALGORITHMS, **REGRESSION_ALGORITHMS}
        if request.algorithm not in all_algorithms:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm. Available: {list(all_algorithms.keys())}"
            )
         # Prepare data
        X = df[request.feature_columns].copy()
        y = df[request.target_column].copy()
        
        mask = y.notna() & X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        if len(X) == 0:
            raise HTTPException(status_code=400, detail="No valid data after removing missing target values")
        
        # For numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns 
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        # Handle categorical columns (fit LabelEncoders)
        categorical_cols = X.select_dtypes(include=['object']).columns
        feature_encoders.clear()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            feature_encoders[col] = le
        # Store global variables
        feature_columns = request.feature_columns
        target_column = request.target_column
        task_type = request.task_type
        
        # Get number of classes for classification
        n_classes = None
        if request.task_type == "classification":
            n_classes = y.nunique()
            if n_classes < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Classification requires at least 2 classes in target variable"
                )
        
        # Validate task type matches target
        if request.task_type == "classification":
            unique_values = y.nunique()
            if unique_values > 20:  # Arbitrary threshold
                logger.warning(f"Target has {unique_values} unique values. Consider regression instead.")
        
        # Handle categorical target for classification
        current_label_encoder = None
        if request.task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            current_label_encoder = LabelEncoder()
            y_encoded = current_label_encoder.fit_transform(y.astype(str))
        else:
            y_encoded = y
        X_train, X_test, y_train, y_test    = None, None, None, None
        # Split data with proper stratification handling
        stratify_param = None 
        if request.task_type == "classification":
            # Only stratify if we have enough samples per class
            class_counts = pd.Series(y_encoded).value_counts()
            min_class_count = class_counts.min()
            if min_class_count >= 2 and len(X) > n_classes * 2:
                stratify_param = y_encoded
        
        # try:
            X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, 
                    test_size=0.2, 
                    random_state=42,
                    stratify=stratify_param
                )  
        if request.task_type =="regression":
            X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, 
                    test_size=0.2, 
                    random_state=42
                )   
        
        current_scaler = None
        if request.algorithm in SCALING_ALGORITHMS:
            
            current_scaler = StandardScaler()
            X_train_scaled = current_scaler.fit_transform(X_train)
            X_test_scaled = current_scaler.transform(X_test)
            X_train_final, X_test_final = X_train_scaled, X_test_scaled
        else:
            X_train_final, X_test_final = X_train.values, X_test.values
        
        # Initialize and train model
        if request.task_type == "classification":
            if request.algorithm not in CLASSIFICATION_ALGORITHMS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Algorithm '{request.algorithm}' not available for classification"
                )
            current_model = CLASSIFICATION_ALGORITHMS[request.algorithm]()
        else:
            if request.algorithm not in REGRESSION_ALGORITHMS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Algorithm '{request.algorithm}' not available for regression"
                )
            current_model = REGRESSION_ALGORITHMS[request.algorithm]()
        
        # Adjust KNN if necessary
        if request.algorithm in ['knn_clf', 'knn_reg']:
            max_neighbors = min(5, len(X_train) - 1) if len(X_train) > 1 else 1
            if hasattr(current_model, 'n_neighbors'):
                current_model.n_neighbors = max(1, max_neighbors) 
        
        # Train model
        logger.info(f"Training {request.algorithm} for {request.task_type}")
        current_model.fit(X_train_final, y_train) 
        
        # Store sample for explainability
        global model_explainer, X_train_sample
        X_train_sample = X_train_final[:100]  # Store sample for SHAP
        
        # # Initialize explainer for tree-based models
        # try:
        #     if 'tree' in request.algorithm or 'forest' in request.algorithm or 'xgboost' in request.algorithm:
        #         model_explainer = shap.TreeExplainer(current_model)
        #     else:
        #         model_explainer = shap.Explainer(current_model, X_train_sample)
        # except Exception as e:
        #     logger.warning(f"Could not initialize explainer: {e}")
        #     model_explainer = None
        
        # Make predictions
        y_pred = current_model.predict(X_test_final)
        
        # Calculate metrics
        if request.task_type == "classification":
            # Handle potential issues with averaging
            avg_method = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
            recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # Cross-validation score with error handling
            try:
                if len(X_train_final) > 3:  # Need at least 3 samples for CV
                    cv_scores = cross_val_score(current_model, X_train_final, y_train, cv=min(3, len(X_train_final)), scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = accuracy  # Use training accuracy as fallback
                    cv_std = 0.0
            except Exception as cv_error:
                logger.warning(f"Cross-validation failed: {str(cv_error)}")
                cv_mean = accuracy
                cv_std = 0.0
            
            model_metrics = {
                "accuracy": f"{accuracy:.3f}",
                "precision": f"{precision:.3f}",
                "recall": f"{recall:.3f}",
                "f1Score": f"{f1:.3f}",
                "confusionMatrix": cm.tolist(),
                "crossValidation": {
                    "mean": f"{cv_mean:.3f}",
                    "std": f"{cv_std:.3f}"
                }
            } 
        else: 
            rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Cross-validation score with error handling
            try:
                if len(X_train_final) > 3:
                    cv_scores = cross_val_score(current_model, X_train_final, y_train, cv=min(3, len(X_train_final)), scoring='r2')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = r2
                    cv_std = 0.0
            except Exception as cv_error:
                logger.warning(f"Cross-validation failed: {str(cv_error)}")
                cv_mean = r2
                cv_std = 0.0
            
            model_metrics = {
                "rmse": f"{rmse:.4f}",
                "mae": f"{mae:.4f}",
                "r2Score": f"{r2:.3f}",
                "mse": f"{mse:.5f}",
                "crossValidation": {
                    "mean": f"{cv_mean:.3f}",
                    "std": f"{cv_std:.3f}"
                }
            }
        
        logger.info(f"Model trained successfully. Metrics: {model_metrics}")
        
         # Store model in dictionary with unique ID
        model_id = str(uuid.uuid4())
        trained_models[model_id] = {
            "model": current_model,
            "scaler": current_scaler,
            "label_encoder": current_label_encoder,
            "feature_encoders": feature_encoders,
            "task_type": request.task_type,
            "algorithm": request.algorithm,
            "metrics": model_metrics,
            "features": feature_columns,
            "target": target_column
        }

        return {
            "message": "Model trained successfully",
            "model_id": model_id,
            "metrics": model_metrics,
            "algorithm": request.algorithm,
            "task_type": request.task_type,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features_used": len(request.feature_columns)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    """Make prediction using trained model"""
    global current_model, current_scaler, current_label_encoder, feature_columns, task_type
    
    try:
        if current_model is None:
            raise HTTPException(status_code=400, detail="No trained model available. Please train a model first.")
        
        # Validate input features
        missing_features = [col for col in feature_columns if col not in request.input_data]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
        
        # Prepare input data
        input_df = pd.DataFrame([request.input_data])
        X = input_df[feature_columns].copy()
        
        # Handle missing values (use 0 for simplicity in prediction)
        X = X.fillna(0)
        
        # Encode categorical columns using saved encoders
        for col in X.select_dtypes(include=['object']).columns:
            if col in feature_encoders:
                le = feature_encoders[col]
                X[col] = X[col].map(
                    lambda v: le.transform([v])[0] if v in le.classes_ else -1
                )
            else:
                X[col] = -1

        
        # Apply scaling if used during training
        if current_scaler is not None:
            X_scaled = current_scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make prediction
        prediction = current_model.predict(X_scaled)[0]
        
        # Generate explanations if requested
        feature_impact = None
        
        
        if task_type == "classification" and hasattr(current_model, 'predict_proba'):
            # Classification with probabilities
            probabilities = current_model.predict_proba(X_scaled)[0]
            
            # Handle label encoding
            if current_label_encoder is not None:
                classes = current_label_encoder.classes_
                predicted_class = current_label_encoder.inverse_transform([int(prediction)])[0]
            else:
                classes = current_model.classes_ if hasattr(current_model, 'classes_') else [str(prediction)]
                predicted_class = prediction
            
            max_prob = max(probabilities) if len(probabilities) > 0 else 0.0
            class_probs = [
                {"class": str(cls), "probability": f"{prob * 100:.1f}"}
                for cls, prob in zip(classes, probabilities)
            ]
            
            return {
                "predictedClass": str(predicted_class),
                "confidence": f"{max_prob * 100:.1f}%",
                "probabilities": class_probs,
                "prediction_type": "classification"
            }
        else:
            # Regression prediction
            # Estimate confidence based on model type
            if hasattr(current_model, 'estimators_') and len(current_model.estimators_) > 1:
                # For ensemble methods, calculate prediction variance
                try:
                    predictions = [est.predict(X_scaled)[0] for est in current_model.estimators_[:min(10, len(current_model.estimators_))]]
                    variance = np.var(predictions)
                    confidence = max(70, min(95, 90 - variance * 0.1))
                except:
                    confidence = 85.0
            else:
                confidence = 85.0
            
            return {
                "predictedValue": f"{prediction:.2f}",
                "confidence": f"{confidence:.1f}%",
                "prediction_type": "regression",
                "feature_impact": feature_impact
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-status")
async def get_model_status():
    """Get current model status and information"""
    global current_model, current_dataset, feature_columns, target_column, task_type, model_metrics
    
    return {
        "model_trained": current_model is not None,
        "dataset_uploaded": current_dataset is not None,
        "dataset_shape": current_dataset.shape if current_dataset is not None else None,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "task_type": task_type,
        "model_type": type(current_model).__name__ if current_model is not None else None,
        "scaler_used": current_scaler is not None,
        "label_encoder_used": current_label_encoder is not None,
        "metrics": model_metrics if model_metrics else None
    }

@app.delete("/reset")
async def reset_model():
    """Reset all model states and clear dataset"""
    global current_model, current_scaler, current_label_encoder, current_dataset
    global feature_columns, target_column, task_type, model_metrics
    
    current_model = None
    current_scaler = None
    current_label_encoder = None
    current_dataset = None
    feature_columns = []
    target_column = ""
    task_type = ""
    model_metrics = {}
    
    logger.info("Model state reset successfully")
    return {"message": "Model state reset successfully"}

@app.get("/dataset-info")
async def get_dataset_info():
    """Get detailed information about the current dataset"""
    global current_dataset
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    
    df = current_dataset
    
    # Basic info
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    }
    
    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        info["categorical_summary"] = {
            col: {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
            for col in categorical_cols
        }
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



@app.get("/download-encoders")
async def download_encoders():
    """Download label encoders as pickle file"""
    global current_label_encoder, feature_encoders, feature_columns, target_column
    
    if not feature_encoders and current_label_encoder is None:
        raise HTTPException(status_code=400, detail="No encoders available")
    
    try:
        encoders_data = {
            "feature_encoders": feature_encoders,
            "label_encoder": current_label_encoder,
            "feature_columns": feature_columns,
            "target_column": target_column
        }
        
        buffer = io.BytesIO()
        joblib.dump(encoders_data, buffer)
        buffer.seek(0)
        
        def generate():
            yield buffer.read()
            buffer.close()
        
        return StreamingResponse(
            generate(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=encoders.joblib"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serialize encoders: {str(e)}")
    
    
@app.get("/download-model/{model_id}")
async def download_model(model_id: str):
    """Download trained model as joblib file"""
    model_entry = trained_models.get(model_id)
    if model_entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        buffer = io.BytesIO()
        joblib.dump(model_entry["model"], buffer)
        buffer.seek(0)
        
        filename = f"model_{model_entry['task_type']}_{model_entry['algorithm']}.joblib"
        
        def generate():
            yield buffer.read()
            buffer.close()
        
        return StreamingResponse(
            generate(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serialize model: {str(e)}")

@app.get("/model-explainability")
async def get_model_explainability():
    """Get global feature importance"""
    global current_model, X_train_sample, feature_columns
    
    if current_model is None or X_train_sample is None:
        raise HTTPException(status_code=400, detail="No trained model or training data available")
    
    try:
        # Get feature importance
        if hasattr(current_model, 'feature_importances_'):
            # Tree-based models
            importances = current_model.feature_importances_
        else:
            # Use permutation importance for other models
            perm_importance = permutation_importance(current_model, X_train_sample, 
                                                   current_model.predict(X_train_sample), 
                                                   n_repeats=5, random_state=42)
            importances = perm_importance.importances_mean
        
        # Create feature importance data
        feature_importance = [
            {"feature": feature_columns[i], "importance": float(importances[i])}
            for i in range(len(feature_columns))
        ]
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "feature_importance": feature_importance,
            "model_type": type(current_model).__name__,
            "explanation_method": "feature_importances" if hasattr(current_model, 'feature_importances_') else "permutation_importance"
        }
        
    except Exception as e:
        logger.error(f"Explainability error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate explanations: {str(e)}")
    
@app.post("/train-all-models")
async def train_all_models(request: Trainingall):
    """Train all algorithms for the given task, using the same preprocessing and metrics flow as single model training."""
    global current_dataset, feature_columns, target_column, task_type
    global feature_encoders, current_label_encoder, X_train_sample
    algo_result.clear()
    
    try:
        # Ensure dataset exists
        if current_dataset is None:
            if os.path.exists(UPLOAD_PATH):
                current_dataset = pd.read_csv(UPLOAD_PATH)
            else:
                raise HTTPException(status_code=400, detail="No dataset uploaded.")
        
        df = current_dataset.copy()
        
        # Validate columns
        missing_features = [col for col in request.feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing feature columns: {missing_features}")
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")
        
        # Prepare data
        X = df[request.feature_columns].copy()
        y = df[request.target_column].copy()
        mask = y.notna() & X.notna().all(axis=1)
        X, y = X[mask], y[mask]
        if len(X) == 0:
            raise HTTPException(status_code=400, detail="No valid data after removing missing values")
        
        # Numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        # Categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        feature_encoders.clear()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            feature_encoders[col] = le
        
        feature_columns = request.feature_columns
        target_column = request.target_column
        task_type = request.task_type
        
        # Encode target if classification and non-numeric
        current_label_encoder = None
        if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            current_label_encoder = LabelEncoder()
            y_encoded = current_label_encoder.fit_transform(y.astype(str))
        else:
            y_encoded = y
        
        # Train-test split
        stratify_param = None
        if task_type == "classification":
            class_counts = pd.Series(y_encoded).value_counts()
            if class_counts.min() >= 2 and len(X) > pd.Series(y_encoded).nunique() * 2:
                stratify_param = y_encoded

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=stratify_param
        ) if task_type == "classification" else train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Loop through all algorithms
        all_algorithms = CLASSIFICATION_ALGORITHMS if task_type == "classification" else REGRESSION_ALGORITHMS
        for algo_name, algo_func in all_algorithms.items():
            # Scaling if required
            scaler = None
            if algo_name in SCALING_ALGORITHMS:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_train_final, X_test_final = X_train_scaled, X_test_scaled
            else:
                X_train_final, X_test_final = X_train.values, X_test.values
            
            model = algo_func()
            # Adjust KNN
            if algo_name in ['knn_clf', 'knn_reg']:
                model.n_neighbors = max(1, min(5, len(X_train) - 1))
            
            # Train model
            model.fit(X_train_final, y_train)
            X_train_sample = X_train_final[:100]
            
            # Store trained model
            trained_models[algo_name] = {
                'model': model,
                'scaler': scaler,
                'task_type': task_type
            }
            
            # Predictions
            y_pred = model.predict(X_test_final)
            
            # Metrics
            if task_type == "classification":
                avg_method = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                metrics = {
                    "accuracy": f"{accuracy:.3f}",
                    "precision": f"{precision:.3f}",
                    "recall": f"{recall:.3f}",
                    "f1Score": f"{f1:.3f}",
                    "confusionMatrix": cm.tolist()
                }
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {
                    "rmse": f"{rmse:.4f}",
                    "mae": f"{mae:.4f}",
                    "r2Score": f"{r2:.3f}",
                    "mse": f"{mean_squared_error(y_test, y_pred):.5f}"
                }
            
            # ✅ Store each trained model with unique ID
            model_id = str(uuid.uuid4())
            trained_models[model_id] = {
                "model": model,
                "scaler": scaler,
                "label_encoder": current_label_encoder,
                "feature_encoders": feature_encoders,
                "task_type": task_type,
                "algorithm": algo_name,
                "metrics": metrics,
                "features": feature_columns,
                "target": target_column
            }

            # Save metrics in algo_result for frontend display
            algo_result[algo_name] = metrics
        
        return {
            "message": "All models trained successfully",
            "metrics": algo_result,
            "task_type": task_type,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features_used": len(feature_columns),
            "available_models": list(trained_models.keys())  # ✅ expose model IDs
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Train-all error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training all models failed: {str(e)}")
