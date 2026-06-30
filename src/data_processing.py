import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(uploaded_file):
    try: 
        return pd.read.csv(uploaded_file)
    
    except unicodeDecodeError:
        try:
            uploaded_file.seek(0)  # Reset file pointer to the beginning
            return pd.read.csv(uploaded_file, encoding='windows-1252')
        
        except unicodeDecodeError:
            uploaded_file.seek(0)  
            return pd.read.csv(uploaded_file, encoding='latin1')

def process_features_targets(df, target_column):
    # Separate features and target
    all_columns = df.columns.tolist()
    features_columns = [col for col in all_columns if col != target_column]
    numeric_features = df[features_columns].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) < 2:
        return None, None, "Error: The dataset must contain at least two numeric features for processing."
    
    X = df[numeric_features]
    y = df[target_column]
    
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())
    
    return X, y, None

def split_and_scale_data(X, y, test_size):
    "split the data into training and testing sets, and scale the features"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test