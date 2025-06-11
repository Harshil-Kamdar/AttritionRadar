import os
import yaml
from src.data.processor import DataProcessor
from src.models.attrition_model import AttritionModel
from src.visualization.dashboard import AttritionDashboard
from src.config_utils import load_config

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main application entry point."""
    try:
        # Load configuration
        config = load_config()
        
        print("Processing data...")
        data_processor = DataProcessor()
        processed_data = data_processor.process_data()
        
        print("\nTraining model...")
        model = AttritionModel(config)
        metrics = model.train(processed_data)
        
        print("\nModel Metrics:")
        if metrics['roc_auc'] is not None:
            print(f"ROC AUC Score: {metrics['roc_auc']:.3f}")
        else:
            print("ROC AUC Score: Not defined (only one class present in y_true)")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        print("\nMaking predictions...")
        predictions = model.predict(processed_data)
        
        print("\nStarting dashboard...")
        dashboard = AttritionDashboard()
        dashboard.update_data(predictions)
        dashboard.run_server()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 