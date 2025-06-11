import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
from typing import Tuple, Dict, Any
import joblib
import os

class AttritionModel:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the attrition prediction model."""
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'tenure',
            'performance_rating',
            'months_since_promotion',
            'team_attrition_rate'
        ]
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        # Select features
        features = [
            'tenure',
            'months_since_promotion',
            'team_attrition_rate',
            'performance_rating',
            'tenure_risk',
            'promotion_risk',
            'team_risk',
            'performance_risk'
        ]
        
        X = df[features].copy()
        
        # Create target variable based on risk score
        y = (df['attrition_risk_score'] > 0.5).astype(int)
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the attrition prediction model."""
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['model']['train_test_split'],
            random_state=self.config['model']['random_state']
        )
        
        # Initialize and train model
        self.model = lgb.LGBMClassifier(
            n_estimators=20,
            learning_rate=0.05,
            max_depth=2,
            min_data_in_leaf=2,
            random_state=self.config['model']['random_state'],
            verbose=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {}
        if len(np.unique(y_test)) < 2:
            metrics['roc_auc'] = None
            metrics['classification_report'] = 'Only one class present in y_true. ROC AUC and classification report are not defined.'
        else:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['classification_report'] = classification_report(y_test, y_pred)
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for new data."""
        # Prepare features
        features = [
            'tenure',
            'months_since_promotion',
            'team_attrition_rate',
            'performance_rating',
            'tenure_risk',
            'promotion_risk',
            'team_risk',
            'performance_risk'
        ]
        X = df[features].copy()
        
        # Make predictions
        predictions = pd.DataFrame({
            'employee_id': df['employee_id'],
            'attrition_probability': self.model.predict_proba(X)[:, 1],
            'high_risk': self.model.predict(X)
        })
        
        return predictions
    
    def save_model(self, path: str = "models/attrition_model.pkl") -> None:
        """Save the trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
    
    def load_model(self, path: str = "models/attrition_model.pkl") -> None:
        """Load a trained model and scaler."""
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.feature_columns = saved_data['feature_columns'] 