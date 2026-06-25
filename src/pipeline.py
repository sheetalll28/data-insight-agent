import pandas as pd
import numpy as np
import sys
import os

# Import our from-scratch math engine!
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from outlier_detection import calculate_iqr_bounds

def run_cleaning_pipeline(raw_df):
    """
    Takes a raw, messy pandas DataFrame and runs it through our
    from-scratch mathematical cleaning engines.
    """
    df = raw_df.copy()
    
    # Aggressively cast strings that are actually numbers
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
                
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Outlier Detection & Clipping (Using our from-scratch IQR engine)
    for col in numeric_cols:
        if col in df.columns:
            # Force everything to be a number (turns text like "UNKNOWN" into NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            valid_data = df[col].dropna().values
            if len(valid_data) > 0:
                # Call our custom Math Engine!
                lower, upper = calculate_iqr_bounds(valid_data)
                
                # Clip the insane outliers (e.g., 999 becomes the upper bound)
                df[col] = df[col].clip(lower, upper)
                
    # 2. Imputation
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
            
    # 3. Deduplication
    # We drop exact row duplicates across all columns for generic compatibility
    df = df.drop_duplicates(keep='first')
    
    return df

def calculate_correlation_matrix(df):
    """
    Calculates the Pearson Correlation Matrix entirely from scratch using NumPy.
    Formula: Covariance(X,Y) / (StdDev(X) * StdDev(Y))
    """
    df_encoded = df.copy()
    
    # Auto-encode categorical variables so they are included in the math
    for col in df_encoded.columns:
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            df_encoded[col] = pd.factorize(df_encoded[col])[0]
            
    cols = df_encoded.columns.tolist()
    
    data = df_encoded[cols].values
    n_cols = data.shape[1]
    
    # Initialize an empty matrix
    corr_matrix = np.zeros((n_cols, n_cols))
    
    for i in range(n_cols):
        for j in range(n_cols):
            x = data[:, i]
            y = data[:, j]
            
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            
            numerator = np.sum((x - mean_x) * (y - mean_y))
            denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
            
            if denominator == 0:
                corr_matrix[i, j] = 0.0
            else:
                corr_matrix[i, j] = numerator / denominator
                
    return pd.DataFrame(corr_matrix, index=cols, columns=cols)

