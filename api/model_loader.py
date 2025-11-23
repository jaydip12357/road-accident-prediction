import pickle
import os

def load_local_pickle(path):
    """Load pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

class ModelLoader:
    #def __init__(self, model_dir='models'):
    def __init__(self, model_dir='models2'):  # Changed from 'models' to 'models2'
        """Load all model artifacts from directory."""
        self.model_dir = model_dir
        
        # Load models
        self.model = load_local_pickle(f'{model_dir}/accident_severity_model.pkl')
        self.scaler = load_local_pickle(f'{model_dir}/scaler.pkl')
        self.label_encoders = load_local_pickle(f'{model_dir}/label_encoders.pkl')
        self.feature_names = load_local_pickle(f'{model_dir}/feature_names.pkl')
        self.categorical_cols = load_local_pickle(f'{model_dir}/categorical_cols.pkl')
        
        print("✅ All models loaded successfully!")
