import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import yaml
import os

class DataProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data processor with configuration."""
        self.config = self._load_config(config_path)
        self.hris_data = None
        self.attrition_data = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> None:
        """Load HRIS and attrition data from configured sources."""
        self.hris_data = pd.read_csv(self.config['data_sources']['hris']['path'])
        self.attrition_data = pd.read_csv(self.config['data_sources']['attrition']['path'])
        
    def calculate_tenure_features(self) -> pd.DataFrame:
        """Calculate tenure-related features."""
        df = self.hris_data.copy()
        
        # Convert dates to datetime
        df['last_promotion_date'] = pd.to_datetime(df['last_promotion_date'])
        current_date = pd.Timestamp.now()
        
        # Calculate months since last promotion
        df['months_since_promotion'] = (
            (current_date - df['last_promotion_date']).dt.days / 30
        ).astype(int)
        
        return df
    
    def calculate_team_attrition(self) -> pd.DataFrame:
        """Calculate team-level attrition metrics."""
        # Merge department info into attrition data
        attrition_with_dept = pd.merge(
            self.attrition_data,
            self.hris_data[['employee_id', 'department']],
            on='employee_id',
            how='left'
        )
        # Group by department and calculate attrition rate
        dept_attrition = attrition_with_dept.groupby('department').size()
        dept_total = self.hris_data.groupby('department').size()
        attrition_rate = (dept_attrition / dept_total).fillna(0)
        # Map attrition rate back to employee data
        self.hris_data['team_attrition_rate'] = self.hris_data['department'].map(attrition_rate)
        return self.hris_data
    
    def calculate_risk_score(self) -> pd.DataFrame:
        """Calculate attrition risk score based on configured thresholds."""
        df = self.hris_data.copy()
        
        # Calculate risk factors with continuous scores instead of binary
        df['tenure_risk'] = df['tenure'].apply(lambda x: min(x/24, 1.0))  # Normalize by 24 months
        df['promotion_risk'] = df['months_since_promotion'].apply(lambda x: min(x/12, 1.0))  # Normalize by 12 months
        df['team_risk'] = df['team_attrition_rate'].apply(lambda x: min(x/0.2, 1.0))  # Normalize by 20%
        
        # Add performance risk
        df['performance_risk'] = (5 - df['performance_rating']) / 4  # Invert and normalize 1-5 scale
        
        # Calculate overall risk score with weighted factors
        df['attrition_risk_score'] = (
            df['tenure_risk'] * 0.3 +
            df['promotion_risk'] * 0.3 +
            df['team_risk'] * 0.2 +
            df['performance_risk'] * 0.2
        )
        
        # Ensure some variation in the risk scores
        df['attrition_risk_score'] = df['attrition_risk_score'].apply(
            lambda x: x + np.random.normal(0, 0.1)  # Add small random noise
        ).clip(0, 1)  # Keep scores between 0 and 1
        
        return df
    
    def process_data(self) -> pd.DataFrame:
        """Run the complete data processing pipeline."""
        if self.hris_data is None or self.attrition_data is None:
            self.load_data()
        df = self.calculate_tenure_features()
        self.hris_data = df  # update hris_data with new features
        df = self.calculate_team_attrition()
        df = self.calculate_risk_score()
        return df
    
    def get_high_risk_employees(self, threshold: float = None) -> pd.DataFrame:
        """Get employees with high attrition risk."""
        if threshold is None:
            threshold = self.config['model']['risk_threshold']
            
        df = self.process_data()
        return df[df['attrition_risk_score'] > threshold]
    
    def get_department_risk_summary(self) -> pd.DataFrame:
        """Get risk summary by department."""
        df = self.process_data()
        return df.groupby('department').agg({
            'attrition_risk_score': ['mean', 'std', 'count'],
            'employee_id': 'count'
        }).round(3) 